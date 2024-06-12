import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

class FFTProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = self.load_data()
        if self.data.size > 0:
            self.perform_fft()
            self.save_fft_data()
        else:
            print("データが読み込まれませんでした。")

    def load_data(self):
        try:
            data = np.loadtxt(self.input_file, delimiter=',')
            print(f"Loaded data shape: {data.shape}")
        except Exception as e:
            print(f"データの読み込み中にエラーが発生しました: {e}")
            data = np.array([])
        return data

    def perform_fft(self):
        fft_data = fft(self.data)
        n = len(fft_data)
        self.fft_amplitude = np.abs(fft_data)[:n // 2] * 2 / n
        self.fft_frequency = np.fft.fftfreq(n, d=1/60)[:n // 2]  # Assuming 60 Hz sampling rate

        plt.figure(figsize=(12, 6))
        plt.plot(self.fft_frequency, self.fft_amplitude)
        plt.title('FFT of Normalized Acceleration Data')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()
        print(f"FFT performed and plotted.")

    def save_fft_data(self):
        try:
            fft_output = np.vstack((self.fft_frequency, self.fft_amplitude)).T
            np.savetxt(self.output_file, fft_output, delimiter=",", fmt='%10.5f', header="Frequency,Amplitude", comments='')
            print(f"FFT data saved to {self.output_file}")
        except Exception as e:
            print(f"データの保存中にエラーが発生しました: {e}")

def main():
    UpDown = 'up'  # 上り坂の場合は 'up'、下り坂の場合は 'down' を指定
    input_csv = 'csv\\' +  UpDown + '-normalized_output.csv'  # 読み込むCSVファイルのパスを指定
    output_csv = 'csv\\' +  UpDown + '-fft_output.csv'  # 保存するファイルのベース名を指定

    fft_processor = FFTProcessor(input_csv, output_csv)

if __name__ == '__main__':
    main()
