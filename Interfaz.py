import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QGridLayout, QTabWidget, QComboBox, QSlider
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import librosa
from scipy.fft import fft, fftfreq 
from scipy.fftpack import fftshift
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter 
import soundfile as sf

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        # Set window title and geometry
        self.setWindowTitle("HMI para procesamiento de señales")
        self.setGeometry(100, 100, 600, 400)

        # Set background gradient using stylesheet
        self.setStyleSheet("background-color: rgb(255, 255, 200);")
        
        # Create a centered label for the title
        title_label = QLabel("HMI para procesamiento de señales", self)
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        title_label.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        title_label.setGeometry(50, 100, 500, 100)

        # Create a button to go to the next section
        next_button = QPushButton("Siguiente sección", self)
        next_button.setGeometry(200, 250, 200, 50)
        next_button.clicked.connect(self.next_section)

    def next_section(self):
        # Hide the main window
        self.hide()

        # Show the next section
        self.next_window = NextSection()
        self.next_window.show()


class NextSection(QWidget):

    def __init__(self):
        super().__init__()

        # Set window title and geometry
        self.setWindowTitle("Sección de procesamiento de señales")
        self.setGeometry(100, 100, 800, 600)  # Increase the size of the window

        # Create a tab widget
        self.tab_widget = QTabWidget(self)

        # Create the file loading tab
        self.file_tab = QWidget()
        self.tab_widget.addTab(self.file_tab, "Cargar archivo")

        # Create a button to load an audio file
        load_button = QPushButton("Cargar archivo", self.file_tab)
        load_button.clicked.connect(self.load_file)

        # Create a label to show the status of file loading
        self.status_label = QLabel("", self.file_tab)

        # Create a layout for the file loading tab
        file_layout = QVBoxLayout(self.file_tab)
        file_layout.addWidget(load_button)
        file_layout.addWidget(self.status_label)

        # Create the processing controls tab
        self.processing_tab = QWidget()
        self.tab_widget.addTab(self.processing_tab, "Controles de Procesamiento")

        # Create a button to apply the filter
        self.filter_button = QPushButton("Aplicar Filtro", self.processing_tab)
        self.filter_button.clicked.connect(self.apply_filter)

        # Create a dropdown menu to select the filter type
        self.filter_type = QComboBox(self.processing_tab)
        self.filter_type.addItems(["Pasa-bajas", "Pasa-altas", "Pasa-banda"])

        # Create sliders to adjust the filter parameters
        self.cutoff_frequency = QSlider(Qt.Orientation.Horizontal, self.processing_tab)
        self.cutoff_frequency.setMinimum(1)  # Set the minimum value to 1
        self.cutoff_frequency.setMaximum(11000)
        self.cutoff_frequency.valueChanged.connect(self.update_cutoff_frequency) 
        
        self.cutoff_frequency_label = QLabel(self.processing_tab)

        self.filter_order = QSlider(Qt.Orientation.Horizontal, self.processing_tab)
        self.filter_order.setMinimum(1)
        self.filter_order.setMaximum(10) 
        self.filter_order.valueChanged.connect(self.update_filter_order)

        self.filter_order_label= QLabel(self.processing_tab)

        # Create a layout for the processing controls tab
        processing_layout = QVBoxLayout(self.processing_tab)
        processing_layout.addWidget(self.filter_button)
        processing_layout.addWidget(QLabel("Tipo de filtro"))
        processing_layout.addWidget(self.filter_type)
        processing_layout.addWidget(QLabel("Frecuencia de corte"))
        processing_layout.addWidget(self.cutoff_frequency)
        processing_layout.addWidget(self.cutoff_frequency_label)
        processing_layout.addWidget(QLabel("Orden del filtro"))
        processing_layout.addWidget(self.filter_order)
        processing_layout.addWidget(self.filter_order_label)

        # Create the Fourier transform tab
        self.fourier_tab = QWidget()
        self.tab_widget.addTab(self.fourier_tab, "Transformada de Fourier")

        # Create a button to apply the Fourier transform
        self.transform_button = QPushButton("Aplicar Transformada", self.fourier_tab)
        self.transform_button.clicked.connect(self.apply_transform)

        #Create a figure with 4 subplots 
        self.figure, self.axes = plt.subplots(2,2, figsize = (5,4))
        self.figure.subplots_adjust(hspace=0.5, wspace=0.3)
        self.canvas = FigureCanvas(self.figure)

        # Create a layout for the Fourier transform tab
        fourier_layout = QVBoxLayout(self.fourier_tab)
        fourier_layout.addWidget(self.transform_button)
        fourier_layout.addWidget(self.canvas)

        # Create the export and save tab
        self.export_tab = QWidget()
        self.tab_widget.addTab(self.export_tab, "Exportación y Guardado")

        # Create a button to save the processed signal
        self.save_button = QPushButton("Guardar Resultado", self.export_tab)
        self.save_button.clicked.connect(self.save_result)

        # Create a dropdown menu to select the output format
        self.output_format = QComboBox(self.export_tab)
        self.output_format.addItems([".wav", ".aac", ".mp3"])

        # Create a layout for the export and save tab
        export_layout = QVBoxLayout(self.export_tab)
        export_layout.addWidget(self.save_button)
        export_layout.addWidget(QLabel("Formato de salida"))
        export_layout.addWidget(self.output_format)

    def load_file(self):
        # Open a file dialog and allow the user to select an audio file
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Audio Files (*.wav *.mp3 *.aac )")
        if fileName:
            self.status_label.setText(f"Archivo cargado: {fileName}")
            print(f"File loaded: {fileName}")

            # Load the audio file
            self.audio_data, self.sample_rate = librosa.load(fileName)

            # Clear the previous plots
            for ax in self.axes.flatten():
                ax.cla()

            # Plot the original signal
            self.axes[0,0].plot(self.audio_data)
            self.axes[0,0].set_title("Señal original")

            # Apply your signal processing here and plot the processed signal
            self.apply_filter()  # Call the apply_filter function here

            # Draw the plots
            self.canvas.draw()

    def apply_filter(self):
        # Get the selected filter type
        filter_type = self.filter_type.currentText()

        # Get the cutoff frequency and filter order from the sliders
        cutoff_frequency = self.cutoff_frequency.value()
        filter_order = self.filter_order.value()

        # Apply the selected filter to the audio data
        if filter_type == "Pasa-bajas":
            # Apply a low-pass filter
            self.filtered_data = self.iir_filter(self.audio_data, cutoff_frequency, self.sample_rate)
        elif filter_type == "Pasa-altas":
            # Apply a high-pass filter
            self.filtered_data = self.iir_filter(self.audio_data, cutoff_frequency, self.sample_rate, btype="high")
        elif filter_type == "Pasa-banda":
            # Apply a band-pass filter
            self.filtered_data = self.iir_filter(self.audio_data, [cutoff_frequency - 0.1, cutoff_frequency + 0.1], self.sample_rate, btype="band")

        # Update the processed signal plot
        self.axes[1,0].cla()
        self.axes[1,0].plot(self.filtered_data)
        self.axes[1,0].set_title("Señal procesada")


        # Draw the plots
        self.canvas.draw()

    def update_cutoff_frequency (self, value): 
        self.cutoff_frequency_label.setText(f"Frecuencia de corte: {value} Hz")
    
    def update_filter_order(self, value):
        self.filter_order_label.setText(f"Orden del filtro: {value}")

    def apply_transform(self):
        # Check if the filtered data is defined
        if hasattr(self, 'filtered_data'):
            # Apply the Fourier transform to the filtered data
            self.transformed_data = fft(self.filtered_data)

            #Clear the previous plots
            for ax in self.axes.flatten(): 
                ax.cla()
            
            # Plot the original signal
            self.axes[0, 0].plot(self.audio_data)
            self.axes[0, 0].set_title('Señal original')

            # Plot the Fourier transform of the original signal
            self.axes[0, 1].plot(np.abs(fft(self.audio_data)))
            self.axes[0, 1].set_title('FFT de la señal original')

            # Plot the filtered signal
            self.axes[1, 0].plot(self.filtered_data)
            self.axes[1, 0].set_title('Señal procesada')

            # Plot the Fourier transform of the filtered signal
            self.axes[1, 1].plot(np.abs(self.transformed_data))
            self.axes[1, 1].set_title('FFT de la señal procesada')

            #Draw the plots 
            self.figure.canvas.draw()
        else:
            print("No filtered data to transform. Please load a file and apply a filter first.")

    def save_result(self):
        # Check if the filtered data is defined
        if hasattr(self, 'filtered_data'):
            # Open a file dialog and allow the user to select a location to save the file
            fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                      "Audio Files (*.wav *.acc *.mp3 )")
            if fileName:
                # Save the processed signal to the selected file
                sf.write(fileName + self.output_format.currentText(), self.filtered_data, self.sample_rate)
        else:
            print("No filtered data to save. Please load a file and apply a filter first.")
    def iir_filter(self, signal, f_cutoff, f_sampling, fbf=False, btype="low"): 
        b, a = iirfilter(4, Wn=f_cutoff, fs=f_sampling, btype=btype, ftype="butter")
        if not fbf:
            filtered = lfilter(b,a, signal)
        else: 
            filtered = filtfilt(b,a, signal)
        return filtered 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


