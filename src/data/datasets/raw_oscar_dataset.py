from os import listdir
from os.path import isfile, join

from pyapnea.oscar.oscar_constants import ChannelID
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_loader import load_session
from torch.utils.data import Dataset

from src.data.preparation_tasks import generate_annotations


# TODO : need sliding window : https://discuss.pytorch.org/t/is-there-a-data-datasets-way-to-use-a-sliding-window-over-time-series-data/115702/4
class RawOscarDataset(Dataset):

    def __init__(self, output_type='numpy', limits=None):
        """

        :param output_type: 'numpy' or 'dataframe'
        """
        self.output_type = output_type
        data_path_cpap1 = '../data/raw/ResMed_23192565579/Events'
        data_path_cpap2 = '../data/raw/ResMed_23221085377/Events'
        self.list_files = [{'label': f, 'value': f, 'fullpath': join(data_path_cpap1, f)} for f in
                           listdir(data_path_cpap1)
                           if isfile(join(data_path_cpap1, f))]
        self.list_files.extend(
            [{'label': f, 'value': f, 'fullpath': join(data_path_cpap2, f)} for f in listdir(data_path_cpap2) if
             isfile(join(data_path_cpap2, f))])

        if limits is not None:
            self.list_files = self.list_files[:limits]

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        result = None
        oscar_session_data = load_session(self.list_files[idx]['fullpath'])
        df = event_data_to_dataframe(oscar_session_data, [ChannelID.CPAP_FlowRate.value,
                                                          ChannelID.CPAP_Obstructive.value])
        df.set_index('time_utc', inplace=True)
        df = generate_annotations(df, length_event='10S')

        if self.output_type == 'numpy':
            result = df[['FlowRate']].to_numpy(), df[['Obstructive']].to_numpy()
        if self.output_type == 'dataframe':
            result = df
        return result
