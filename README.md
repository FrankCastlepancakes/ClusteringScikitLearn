# Clustering mit Scikit-Learn

Autoren: Mario Pfob, Marius Harrass

---

## 1. Definition Cluster-Analyse

## 2. Kontext Datensatz

## 3. Daten visualisieren & aufbereiten

Die Autoren *García, Salvador u. a. (2016)* stellen in ihrem Schaubild den Erkentnissgewinn aus Daten als iterierenden Prozess dar (siehe Bild *Knowledge Discovery in Databases - Prozess*). Es wird bei dem Prozess davon ausgegangen, dass die Daten nicht im gewünschten Format vorliegen oder beispielsweise Lücken innerhalb der Daten vorhanden sind. \
Dementsprechend beginnt das Schaubild bei dem Schritt der Problem-Spezifikation: Bezogen auf den vorliegenden Datensatz soll ein konkretes Problem bzw. Frage beantwortet werden.\
Ist eine Problemstellung festgelegt, folgt das Verständis. Damit ist gemeint, dass der Datensatz in seiner rohen Form betrachtet wird, um u.a. notwendige Aufgaben für den folgenden Schritt der Daten-Aufbereitung zu entdecken.\
Bei der Aufbereitung sind mehrere Optionen vorhanden, um die Qualität des Datensatzes zu verbessern. Diese Optionen sind auf der Grafik *Data prepocessing tasks* zu sehen. Des Weiteren geben *García, Salvador u. a. (2016)* mit dem Schaubild *Data reduction approaches* Möglichkeiten zur Reduzierung der **relevaten** Daten, ggf. wird somit ein besseres Ergebnis erziehlt.\
Im Prozess des Data-Mining werden wiederum die nun aufbereiteten Daten explizit mit statistischen Methoden betrachtet, bspw. durch lineare Regression, Klassifizierung, oder wie in diesem Projekt durch **Clustering**.\
Schlussendlich werden bei der Evaluation, der Bewertung, Muster aufgezeigt, bewertet und ggf. verglichen.\
Diese Muster werden dann im letzten Punkt der Ergebnis-Verwertung präsentiert, z.B. in Form von Charts.

![Knowledge Discovery in Databases - Prozess](images/Knowledge_Discovery_Databases.png) \
*Knowledge Discovery in Databases - Prozess* \
*Entnommen aus: García, Salvador u. a. (2016)*

![Data prepocessing tasks](images/Data_Prepocessing_Tasks.png) \
*Data prepocessing tasks* \
*Entnommen aus: García, Salvador u. a. (2016)*

![Data reduction approaches](images/Data_Reduction_Approaches.png) \
*Data reduction approaches* \
*Entnommen aus: García, Salvador u. a. (2016)*

#
Transferiert auf die Vorbereitung in diesem Abschnitt, ist dementsprechend zuerst eine Visualisierung der Daten notwendig. Die Daten können sowohl im ursprünglichen CSV-Format, aber auch als Tabelle oder als Plot (siehe *Plot 1*) betrachtet werden. Daraus leiten sich mehrere zu erledigende Aufgaben ab:

- **Data Cleaning**, bspw. ist es notwendig die deutsche Zahlenschreibweise auf die amerikanische Schreibweise anzupassen
- **Data Integration**, aus den Spalten 'Length', 'Height', 'Width' soll das dazugehörige Volumen berechnet werden
- **Data Normalization**, d.h. alle (metrischen) Spalten sollen auf eine einheitliche Metrik standardisiert werden
- **Feature Selection**, so werden bspw. die Spalten 'Package No' und 'Shipment No' für die Cluster-Analyse nicht benötigt

Die Bearbeitung dieser Aufgaben ist durch die Grafiken *Plot 2* & *Plot 3* und *Tabelle 1* zu nachzuvollziehen.

![Package data before preprocessing](images/Package_Data_Before_Preprocessing.png) \
*Plot 1: Der Packstücke-Datensatz bevor dem Prozess der Daten-Aufbereitung. Die Eigenschaften 'Length' und 'Height' sind als Scatter-Plot visualisiert* \
*Quelle: Eigendarstellung mittels Matplotlib*

![Package data during preprocessing](images/Package_Data_During_Preprocessing.png) \
*Plot 2: Der Packstücke-Datensatz während der Daten-Aufbereitung. Die neue Eigenschaft 'Volumen' wird zusammen mit dem 'Gewicht'  als Scatter-Plot visualisiert* \
*Quelle: Eigendarstellung mittels Matplotlib*

![Package data after preprocessing](images/Package_Data_After_Preprocessing.png) \
*Plot 3: Der Packstücke-Datensatz nach der Daten-Aufbereitung. Die Eigenschaften 'Volumen' und 'Gewicht' sind mit standardisisierten Skalen als Scatter-Plot visualisiert* \
*Quelle: Eigendarstellung mittels Matplotlib*

![Tabellarische_Darstellung_Datenaufbereitung](images/Tabellarische_Darstellung_Datenaufbereitung.png) \
*Tabelle 1: Die ersten zehn Zeilen des Datensatzes vor und nach der Aufbereitung*
*Quelle: Eigendarstellung durch Pandas*

## 4. Cluster-Analyse: KMeans

## 5. Cluster-Analyse: Hierarchisch

## 6. Fazit

## 7. Kritische Reflexion
