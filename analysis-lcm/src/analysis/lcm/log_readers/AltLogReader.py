import numpy as np
from aspn23_lcm import measurement_barometer
from aspn23_xtensor import to_seconds

from analysis.lcm.data import AltData, DataType
from analysis.lcm.measurements import get_altitude, get_temp

from .LogReader import LogReader


class AltLogReader(LogReader[AltData]):
    # Map temperature channels to paired altitude channels
    temp_to_alt: dict[str, str] = {}
    alt_to_temp: dict[str, str] = {}

    def __init__(
        self,
        logfile: str,
        desired_types: tuple,
        save_all: bool,
        config_file: str,
    ):
        super().__init__(logfile, AltData, desired_types, save_all, config_file)
        self.log_data.truth_channel = self.config.get('truth_pva_channel', None)
        self.log_data.data[self.log_data.truth_channel] = self.new_data(
            self.log_data.truth_channel
        )

    def new_data(self, label) -> DataType:
        bias = 0.0
        if label in self.config:
            channel_config: dict = self.config[label]
            if 'bias' in channel_config:
                bias = channel_config['bias']
            if 'temperature_channel' in channel_config:
                temperature_channel = channel_config['temperature_channel']
                self.temp_to_alt[temperature_channel] = label
                self.alt_to_temp[label] = temperature_channel
                self.channels_to_ignore.discard(temperature_channel)
                self.channels_to_keep.add(temperature_channel)

        return AltData(label, bias)

    def save_msg(self, channel, time, aspn_msg):
        temp, _ = get_temp(aspn_msg)
        if temp is not None:
            alt_channel = self.temp_to_alt[channel]
            self.log_data.data[alt_channel].temp_time.append(time)
            self.log_data.data[alt_channel].temp.append(temp + 273.15)
            return

        if channel in self.alt_to_temp:
            if not self.log_data.data[channel].temp:
                return
        temp = 288.15
        # temp = self.log_data.data[channel].temp[-1] if self.log_data.data[channel].temp else 288.15

        if isinstance(aspn_msg, measurement_barometer):
            if (
                self.log_data.data[channel].ref_pressure == 101325.0
                and self.log_data.data[self.log_data.truth_channel].alt
            ):
                self.log_data.data[channel].ref_pressure = aspn_msg.pressure
                self.log_data.data[channel].ref_alt = 0.0

        alt, sigma = get_altitude(
            aspn_msg,
            self.log_data.data[channel].bias,
            temp,
            self.log_data.data[channel].ref_pressure,
            self.log_data.data[channel].ref_alt,
        )

        self.log_data.data[channel].time.append(time)
        self.log_data.data[channel].alt.append(alt)
        self.log_data.data[channel].sigma.append(sigma)

    def postprocess(self):
        # Convert tov to relative time and altitude to numpy array
        for data in self.log_data.data.values():
            data.time = np.array([to_seconds(t - self.log_data.t0) for t in data.time])
            data.alt = np.array(data.alt)
            data.sigma = np.array(data.sigma)
            data.temp_time = np.array(
                [to_seconds(t - self.log_data.t0) for t in data.temp_time]
            )
            data.temp = np.array(data.temp)
