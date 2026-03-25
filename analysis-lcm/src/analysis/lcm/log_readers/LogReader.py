from abc import ABC
from typing import Generic

from lcm import Event, EventLog
from tomlkit.toml_file import TOMLDocument, TOMLFile
from tqdm import tqdm

from analysis.lcm.data import DataType, LogData
from analysis.lcm.measurements import decode_aspn_lcm_msg, is_type


class LogReader(ABC, Generic[DataType]):
    log_data: LogData[DataType]
    data_type: type[DataType]
    channels_to_ignore: set[str]
    channels_to_keep: set[str]
    save_all: bool
    desired_types: tuple
    log: EventLog

    def __init__(
        self,
        logfile: str,
        data_type: type[DataType],
        desired_types: tuple,
        save_all: bool = False,
        config_file: str = '',
    ):
        self.data_type = data_type
        self.channels_to_ignore = set()
        self.channels_to_keep = set()
        self.desired_types = desired_types
        self.save_all = save_all
        self.log = EventLog(logfile, 'r')
        self.config = TOMLFile(config_file).read() if config_file else TOMLDocument()
        self.log_data = LogData(logfile)

    def keep_msg(self, msg: Event) -> bool:
        """Return true if we want to keep this message."""
        # Keep if already kept message on this channel before
        if msg.channel in self.log_data.data or msg.channel in self.channels_to_keep:
            return True

        # Don't keep if already ignoring messages on this channel
        if msg.channel in self.channels_to_ignore:
            return False

        # Don't keep if not of desired type
        if not is_type(msg, *self.desired_types):
            self.channels_to_ignore.add(msg.channel)
            return False

        # Keep if saving all messages of desired types
        if self.save_all:
            self.log_data.data[msg.channel] = self.new_data(msg.channel)
            return True

        # Get user input to decide whether to save messages on this channel
        usr_input = input(f'Found message on channel {msg.channel}. Use it? [y/n]')
        if usr_input == 'y':
            self.log_data.data[msg.channel] = self.new_data(msg.channel)
            return True
        else:
            self.channels_to_ignore.add(msg.channel)
            return False

    def new_data(self, label: str) -> DataType:
        """Create new instance of Data."""
        return self.data_type(label)

    def save_msg(self, channel, time, aspn_msg):
        """Save data from aspn_msg to Data object stored at self.data[channel]."""
        self.log_data.data[channel].add_data(time, aspn_msg)

    def postprocess(self):
        """Perform some optional postprocessing on the ingested data before it's used."""
        pass

    def read_log(self) -> LogData[DataType]:
        print('Reading measurements from log...')
        msg: Event
        log_size = self.log.size()
        # TODO: would be nice if LCM log exposed num_events field so that we could just do 'for msg in tqdm(log, total=log.num_events)'.
        progressbar = tqdm(total=log_size, unit='B', unit_scale=True)
        fpos = self.log.tell()
        for msg in self.log:
            new_fpos = self.log.tell()
            progressbar.update(new_fpos - fpos)
            fpos = new_fpos

            if self.keep_msg(msg):
                t, aspn_msg = decode_aspn_lcm_msg(msg)
                if self.log_data.t0 is None:
                    self.log_data.t0 = t  # set initial time
                self.save_msg(msg.channel, t, aspn_msg)
        progressbar.close()

        self.postprocess()

        return self.log_data
