import os.path

import ssl

import pandas as pd

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import seaborn as sbs
import sqlite3

from joblib import dump, load
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from screeninfo import get_monitors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

# panda data setup
# =======================================================================================================================
dataFile = 'pliktextowy.txt'
modelFile = 'model.joblib'

ssl._create_default_https_context = ssl._create_unverified_context

url = ""
headers = []

file1 = open(dataFile, 'r')
lines = file1.readlines()
counter = 0
for line in lines:
    if counter == 0:
        url = line.rstrip()
    else:
        headers.append(line.rstrip())
    counter += 1

try:
    conn = sqlite3.connect('wines_dominik_nykiel')
    cursor = conn.cursor()
    cursor.execute("SELECT * from DataTable")

    pandaData = pd.DataFrame(cursor.fetchall(), columns=headers)

except sqlite3.Error as e:
    print("No data in database, creating new")
    pandaData = pd.read_csv(url, names=headers)

    cursor.execute(
        'CREATE TABLE IF NOT EXISTS DataTable (TypeOf number ,Alcohol number, Malic_acid number, Ash number, Alcalinity_of_ash number,Magnesium number,Total_phenols number,Flavanoids number,Nonflavanoid_phenols number,Proanthocyanins number,Color_intensity number,Hue number,OD280_OD315_of_diluted_wines number, Proline number)')

    pandaData.to_sql('DataTable', conn, if_exists='replace', index=False)

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()

X = pandaData.iloc[:, 1:].values
y = pandaData.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023)

if os.path.exists(modelFile):
    knn = load(modelFile)
else:
    knn = KNeighborsClassifier(n_neighbors=5)
# end of panda data setup
# =======================================================================================================================

# window initialize
# =======================================================================================================================

root = tk.Tk()
root.title("kNN classifier")

screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height
root.geometry(f"{int(screen_width / 2) + 350}x{int(screen_height / 2)}")

left_frame = tk.Frame(root, borderwidth=4, relief="ridge", width=int(screen_width / 8), height=int(screen_width / 4))
left_frame.pack(side="left", padx=10, pady=10)
left_frame.pack_propagate(False)

grade_frame = tk.Frame(root, width=int(screen_width / 4), height=left_frame["height"], borderwidth=4, relief="ridge")
grade_frame.pack(side="right", padx=10, pady=10)
grade_frame.pack_propagate(False)

result_frame = tk.Frame(root, width=int(screen_width / 4), height=left_frame["height"], borderwidth=4, relief="ridge")
result_frame.pack(side="right", padx=10, pady=10)
result_frame.pack_propagate(False)

gradeText = tk.Text(grade_frame, height=100,
                    width=80,
                    bg="light yellow")

resultText = tk.Text(result_frame, height=100,
                     width=200,
                     bg="light cyan")

gradeLabel = tk.Label(grade_frame, text="Ocena modelu")
gradeLabel.pack()
gradeText.pack()
resultLabel = tk.Label(result_frame, text="Wynik dla rekordu")
resultLabel.pack()
resultText.pack()


# end of window initialize
# =======================================================================================================================

# data display functions
# =======================================================================================================================
def fetch_data():
    try:
        myconn = sqlite3.connect('wines_dominik_nykiel')
        mycursor = myconn.cursor()
        mycursor.execute("SELECT * FROM DataTable")
        result = mycursor.fetchall()
        return result
    except sqlite3.Error as exc:
        print(f"Error: {exc}")
    finally:
        if mycursor:
            mycursor.close()
        if myconn:
            myconn.close()


def displaydatawindow(dataToModify):
    displaywindow = tk.Toplevel(root)
    treeview = ttk.Treeview(displaywindow)
    treeview["columns"] = headers
    treeview.column("#0", width=0)

    for i in range(0, len(headers)):
        treeview.heading(headers[i], text=headers[i])
    treeview.pack(fill='x')

    h = ttk.Scrollbar(displaywindow, orient='horizontal', command=treeview.xview)
    h.pack(side='bottom', fill='x')
    treeview.configure(xscrollcommand=h.set)

    def load_data():
        data = fetch_data()
        treeview.delete(*treeview.get_children())
        for row in data:
            treeview.insert("", "end", values=(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                                               row[9], row[10], row[11], row[12], row[13]))

    def open_insert_window():

        new_window = tk.Toplevel(displaywindow)
        new_window.title("Dodaj nowy rekord")

        type_label = ttk.Label(new_window, text="TypeOf")
        type_label.pack()
        type_entry = ttk.Entry(new_window)
        type_entry.pack()

        alcohol_label = ttk.Label(new_window, text="Alcohol")
        alcohol_label.pack()
        alcohol_entry = ttk.Entry(new_window)
        alcohol_entry.pack()

        acid_label = ttk.Label(new_window, text="Malic acid")
        acid_label.pack()
        acid_entry = ttk.Entry(new_window)
        acid_entry.pack()

        ash_label = ttk.Label(new_window, text="Ash")
        ash_label.pack()
        ash_entry = ttk.Entry(new_window)
        ash_entry.pack()

        alcaline_label = ttk.Label(new_window, text="Alcalinity of ash")
        alcaline_label.pack()
        alcaline_entry = ttk.Entry(new_window)
        alcaline_entry.pack()

        magnesium_label = ttk.Label(new_window, text="Magnesium")
        magnesium_label.pack()
        magnesium_entry = ttk.Entry(new_window)
        magnesium_entry.pack()

        phenols_label = ttk.Label(new_window, text="Total_phenols")
        phenols_label.pack()
        phenols_entry = ttk.Entry(new_window)
        phenols_entry.pack()

        flavanoid_label = ttk.Label(new_window, text="Flavanoids")
        flavanoid_label.pack()
        flavanoid_entry = ttk.Entry(new_window)
        flavanoid_entry.pack()

        nonflavanoid_label = ttk.Label(new_window, text="Nonflavanoid phenols")
        nonflavanoid_label.pack()
        nonflavanoid_entry = ttk.Entry(new_window)
        nonflavanoid_entry.pack()

        proanth_label = ttk.Label(new_window, text="Proanthocyanins")
        proanth_label.pack()
        proanth_entry = ttk.Entry(new_window)
        proanth_entry.pack()

        color_label = ttk.Label(new_window, text="Color_intensity")
        color_label.pack()
        color_entry = ttk.Entry(new_window)
        color_entry.pack()

        hue_label = ttk.Label(new_window, text="Hue")
        hue_label.pack()
        hue_entry = ttk.Entry(new_window)
        hue_entry.pack()

        dilute_label = ttk.Label(new_window, text="OD280_OD315_of_diluted_wines")
        dilute_label.pack()
        dilute_entry = ttk.Entry(new_window)
        dilute_entry.pack()

        proline_label = ttk.Label(new_window, text="Proline")
        proline_label.pack()
        proline_entry = ttk.Entry(new_window)
        proline_entry.pack()

        def add_new():

            new_type = type_entry.get()
            new_alcohol = alcohol_entry.get()
            new_acid = acid_entry.get()
            new_ash = ash_entry.get()
            new_alcaline = alcaline_entry.get()
            new_magnesium = magnesium_entry.get()
            new_phenols = phenols_entry.get()
            new_flavanoid = flavanoid_entry.get()
            new_nonflava = nonflavanoid_entry.get()
            new_proanth = proanth_entry.get()
            new_color = color_entry.get()
            new_hue = hue_entry.get()
            new_dilute = dilute_entry.get()
            new_proline = proline_entry.get()
            try:
                connfunc = sqlite3.connect('wines_dominik_nykiel')
                cursorfunc = connfunc.cursor()
                sql = "INSERT INTO DataTable (TypeOf, Alcohol, Malic_acid, Ash ,Alcalinity_of_ash ,Magnesium, Total_phenols, Flavanoids, Nonflavanoid_phenols, Proanthocyanins, Color_intensity, Hue, OD280_OD315_of_diluted_wines, Proline) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                params = (new_type, new_alcohol, new_acid, new_ash, new_alcaline, new_magnesium, new_phenols,
                          new_flavanoid, new_nonflava, new_proanth, new_color, new_hue, new_dilute, new_proline)
                cursorfunc.execute(sql, params)
                connfunc.commit()
            except sqlite3.Error as e:
                print(f"Error: {e}")
            finally:
                if cursorfunc:
                    cursorfunc.close()
                if connfunc:
                    connfunc.close()

            load_data()
            new_window.destroy()

        update_button = tk.Button(new_window, text="Dodaj rekord", command=add_new)
        update_button.pack()

    def save_data(dataToSave):
        dataToSave = pd.DataFrame(fetch_data(), columns=headers)
        print(dataToSave.shape)

    add_button = tk.Button(displaywindow, text="Dodaj nowy rekord", command=open_insert_window)
    add_button.pack(side='left')

    save_button = tk.Button(displaywindow, text="Zapisz rekord", command=lambda: save_data(dataToModify))
    save_button.pack(side='left')

    load_data()


def show_plot(data):
    new_window = tk.Toplevel(root)

    figure = Figure(figsize=(6, 6))
    ax = figure.subplots()

    sbs.scatterplot(x=data['Alcohol'], y=data['Flavanoids'], hue=data['TypeOf'], ax=ax)

    canvas = FigureCanvasTkAgg(figure, master=new_window)

    canvas.draw()

    canvas.get_tk_widget().pack()


    toolbar = NavigationToolbar2Tk(canvas,
                                   new_window)
    toolbar.update()

    canvas.get_tk_widget().pack()


# end of display functions
# =======================================================================================================================

# actual model functions
# =======================================================================================================================

def trainnewmodel(currentModel, testset_X, testset_Y, trainset_X, trainset_Y):
    trainset_X, testset_X, trainset_Y, testset_Y = train_test_split(X, y, test_size=0.25, random_state=2023)
    print(testset_X)
    model = KNeighborsClassifier(n_neighbors=5)
    print(trainset_X.shape)
    print(trainset_Y.shape)
    model.fit(trainset_X, trainset_Y)
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'metric': ['euclidean', 'manhattan']
    }
    grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, random_state=2023, shuffle=True))
    grid_search.fit(trainset_X, trainset_Y)
    currentModel = grid_search.best_estimator_
    print(currentModel)
    dump(grid_search.best_estimator_, modelFile)


def test_new_model(testset_X, testset_Y, trainset_X, trainset_Y, kNN_model):
    gradeText.delete("1.0", "end")

    best_predict = kNN_model.predict(testset_X)

    gradeText.insert(tk.END, f"Dokładność modelu na zbiorze testowym:  {accuracy_score(testset_Y, best_predict)} \n")

    best_predict_train = kNN_model.predict(trainset_X)

    gradeText.insert(tk.END,
                     f"Dokładność modelu na zbiorze treningowym:  {accuracy_score(trainset_Y, best_predict_train)} \n")

    cm_train = confusion_matrix(trainset_Y, best_predict_train)
    gradeText.insert(tk.END, f"Macierz pomyłek dla zbioru treningowego:\n  {cm_train} \n")

    report = classification_report(trainset_Y, best_predict_train)
    gradeText.insert(tk.END, report)


def test_new_recordwindow(model):
    new_window = tk.Toplevel(root)
    new_window.title("Przetestuj rekord")

    alcohol_label = ttk.Label(new_window, text="Alcohol")
    alcohol_label.pack()
    alcohol_entry = ttk.Entry(new_window)
    alcohol_entry.pack()

    acid_label = ttk.Label(new_window, text="Malic acid")
    acid_label.pack()
    acid_entry = ttk.Entry(new_window)
    acid_entry.pack()

    ash_label = ttk.Label(new_window, text="Ash")
    ash_label.pack()
    ash_entry = ttk.Entry(new_window)
    ash_entry.pack()

    alcaline_label = ttk.Label(new_window, text="Alcalinity of ash")
    alcaline_label.pack()
    alcaline_entry = ttk.Entry(new_window)
    alcaline_entry.pack()

    magnesium_label = ttk.Label(new_window, text="Magnesium")
    magnesium_label.pack()
    magnesium_entry = ttk.Entry(new_window)
    magnesium_entry.pack()

    phenols_label = ttk.Label(new_window, text="Total_phenols")
    phenols_label.pack()
    phenols_entry = ttk.Entry(new_window)
    phenols_entry.pack()

    flavanoid_label = ttk.Label(new_window, text="Flavanoids")
    flavanoid_label.pack()
    flavanoid_entry = ttk.Entry(new_window)
    flavanoid_entry.pack()

    nonflavanoid_label = ttk.Label(new_window, text="Nonflavanoid phenols")
    nonflavanoid_label.pack()
    nonflavanoid_entry = ttk.Entry(new_window)
    nonflavanoid_entry.pack()

    proanth_label = ttk.Label(new_window, text="Proanthocyanins")
    proanth_label.pack()
    proanth_entry = ttk.Entry(new_window)
    proanth_entry.pack()

    color_label = ttk.Label(new_window, text="Color_intensity")
    color_label.pack()
    color_entry = ttk.Entry(new_window)
    color_entry.pack()

    hue_label = ttk.Label(new_window, text="Hue")
    hue_label.pack()
    hue_entry = ttk.Entry(new_window)
    hue_entry.pack()

    dilute_label = ttk.Label(new_window, text="OD280_OD315_of_diluted_wines")
    dilute_label.pack()
    dilute_entry = ttk.Entry(new_window)
    dilute_entry.pack()

    proline_label = ttk.Label(new_window, text="Proline")
    proline_label.pack()
    proline_entry = ttk.Entry(new_window)
    proline_entry.pack()

    def test_new_record(currentmodel):
        new_alcohol = alcohol_entry.get()
        new_acid = acid_entry.get()
        new_ash = ash_entry.get()
        new_alcaline = alcaline_entry.get()
        new_magnesium = magnesium_entry.get()
        new_phenols = phenols_entry.get()
        new_flavanoid = flavanoid_entry.get()
        new_nonflava = nonflavanoid_entry.get()
        new_proanth = proanth_entry.get()
        new_color = color_entry.get()
        new_hue = hue_entry.get()
        new_dilute = dilute_entry.get()
        new_proline = proline_entry.get()
        newRecord = [[float(new_alcohol), float(new_acid), float(new_ash), float(new_alcaline), float(new_magnesium),
                      float(new_phenols), float(new_flavanoid),
                      float(new_nonflava), float(new_proanth), float(new_color), float(new_hue), float(new_dilute),
                      float(new_proline)]]
        resultText.insert(tk.END, f"Wynik dla rekordu {newRecord}: \n {currentmodel.predict(newRecord)} \n")
        new_window.destroy()

    make_predict = tk.Button(new_window, text="Klasyfikuj!", command=lambda: test_new_record(model))
    make_predict.pack()


# end of model functions
# =======================================================================================================================


data_button = tk.Button(left_frame, text="Pokaż dane", command=lambda: displaydatawindow(pandaData))
data_button.pack(anchor="w", padx=10, pady=10)

test_button = tk.Button(left_frame, text="Testuj model",
                        command=lambda: test_new_model(X_test, y_test, X_train, y_train, knn))
test_button.pack(anchor="w", padx=10, pady=10)

train_button = tk.Button(left_frame, text="Wytrenuj nowy model",
                         command=lambda: trainnewmodel(knn, X_test, y_test, X_train, y_train))
train_button.pack(anchor="w", padx=10, pady=10)

predict_button = tk.Button(left_frame, text="Wprowadz rekord do klasyfikacji",
                           command=lambda: test_new_recordwindow(knn))
predict_button.pack(anchor="w", padx=10, pady=10)

plot_button = tk.Button(left_frame, text="Pokaż wykres danych", command=lambda: show_plot(pandaData))
plot_button.pack(anchor="w", padx=10, pady=10)
root.mainloop()
