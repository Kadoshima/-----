import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class InData:
    label: int
    InData = np.array([])
    deletelen = 1  # seconds to cut from start and end
    samplingRate = 60  # Hz
    segment_length = 5  # seconds

    def __init__(self, url, label) -> None:
        self.label = label
        with open(url, encoding='utf8', newline='') as f:
            csvreader = csv.reader(f, delimiter=',')
            data = [row for row in csvreader]
            self.InData = np.array(data, dtype=float)
        self.resample_data(self.samplingRate)
        self.trim_data(self.deletelen)

    def resample_data(self, new_rate):
        timestamps = self.InData[:, 0]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        num_samples = int((end_time - start_time) / 1e3 * new_rate)  # タイムスタンプがミリ秒単位の場合
        new_timestamps = np.linspace(start_time, end_time, num_samples)

        interp_x = interp1d(timestamps, self.InData[:, 1], kind='linear')
        interp_y = interp1d(timestamps, self.InData[:, 2], kind='linear')
        interp_z = interp1d(timestamps, self.InData[:, 3], kind='linear')

        self.InData = np.vstack((
            new_timestamps,
            interp_x(new_timestamps),
            interp_y(new_timestamps),
            interp_z(new_timestamps)
        )).T

    def trim_data(self, trim_seconds):
        trim_samples = trim_seconds * self.samplingRate
        if len(self.InData) > 2 * trim_samples:
            self.InData = self.InData[trim_samples:-trim_samples]
        else:
            print("トリムするデータが不足しています")

    def segment_data(self):
        segment_samples = self.segment_length * self.samplingRate
        num_segments = len(self.InData) // segment_samples
        segments = []

        for i in range(num_segments):
            segment = self.InData[i * segment_samples:(i + 1) * segment_samples]
            segments.append(segment)

        return segments

    def save_segments(self, segments, output_dir):
        os.makedirs(output_dir, exist_ok=True)  # 出力ディレクトリが存在しない場合は作成
        for i, segment in enumerate(segments):
            filename = f"{output_dir}/segment_{i}_label_{self.label}.csv"
            np.savetxt(filename, segment, delimiter=",", header="Timestamp,X,Y,Z", comments='', fmt='%10.5f')

    def printData(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.InData[:, 0], self.InData[:, 1], label='X axis')
        plt.plot(self.InData[:, 0], self.InData[:, 2], label='Y axis')
        plt.plot(self.InData[:, 0], self.InData[:, 3], label='Z axis')
        plt.xlabel('Timestamp')
        plt.ylabel('Acceleration')
        plt.title('Acceleration Sensor Data (Resampled at 60Hz)')
        plt.legend()
        plt.grid()
        plt.show()

def __main__():
    data = InData('きつい坂下り\きつい下り坂01-K.csv', 0)
    if data is None:
        print('データの読み込みに失敗しました')
        return

    segments = data.segment_data()
    output_dir = 'output_directory_path'
    data.save_segments(segments, output_dir)

    data.printData()

if __name__ == '__main__':
    __main__()
