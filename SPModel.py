# %%
import numpy as np
import torch
from torch import nn
import math

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%
ReferLen = 1200
QueryLen = 60

weekconvchannel = 32
monthconvchannel = 64
seasonconvchannel = 64

rdconvchannel = 64


# %%
class DateEncoding(nn.Module):
    def __init__(self, d_embed, dropout=0.2, maxlen=ReferLen):
        super(DateEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_embed = d_embed

    def forward(self, embed, dates):
        pe = torch.zeros_like(embed)
        batch_size = dates.shape[0]
        div_term = torch.exp(torch.arange(0, self.d_embed, 2).float() * (-math.log(10000.0)/self.d_embed)).unsqueeze(1).unsqueeze(0)
        div_term = torch.cat([div_term for i in range(batch_size)], dim=0).to(device)
        pe[:, 0::2, :] = torch.sin(torch.bmm(div_term, dates))
        pe[:, 1::2, :] = torch.cos(torch.bmm(div_term, dates))
        embed = embed + torch.autograd.Variable(pe, requires_grad=False)
        return self.dropout(embed)


# %%
class ReferEncoder(nn.Module):
    def __init__(self):
        super(ReferEncoder, self).__init__()

        self.ReLUlayer = nn.LeakyReLU()

        self.dateweekencoding = DateEncoding((8+weekconvchannel+monthconvchannel+seasonconvchannel), dropout=0.1, maxlen=ReferLen)

        self.conv1dday = nn.Conv1d(6, 2, 1, 1, padding=0, bias=True)
        self.BNday = nn.BatchNorm1d(2)

        self.conv1dweek = nn.Conv1d(8, weekconvchannel, 5, stride=1, padding=2, padding_mode='circular', bias=True)
        self.BNweek = nn.BatchNorm1d(weekconvchannel)

        self.datepollingmonth = nn.AvgPool1d(5, stride=5)
        self.datemonthencoding = DateEncoding((monthconvchannel+seasonconvchannel), dropout=0.1, maxlen=ReferLen//5)

        self.conv1dmonth1 = nn.Conv1d(weekconvchannel, monthconvchannel, 10, stride=5, padding=4, padding_mode='circular', bias=True)
        self.BNmonth1 = nn.BatchNorm1d(monthconvchannel)
        self.conv1dmonth2 = nn.Conv1d(monthconvchannel, monthconvchannel, 4, stride=1, padding='same', padding_mode='circular', bias=True)
        self.BNmonth2 = nn.BatchNorm1d(monthconvchannel)

        self.datepoolingseason = nn.AvgPool1d(20, stride=20)
        self.dateseasonencoding = DateEncoding(seasonconvchannel, dropout=0.1, maxlen=ReferLen//20)

        self.conv1dseason1 = nn.Conv1d(monthconvchannel, seasonconvchannel, 8, stride=4, padding=3, padding_mode='circular', bias=True)
        self.BNseason1 = nn.BatchNorm1d(seasonconvchannel)
        self.conv1dseason2 = nn.Conv1d(seasonconvchannel, seasonconvchannel, 3, stride=1, padding=1, padding_mode='circular', bias=True)
        self.BNseason2 = nn.BatchNorm1d(seasonconvchannel)

        self.seasoninterpacgpool = nn.AvgPool1d(9, stride=1, padding=4, count_include_pad=False)
        self.monthinterpavgpool = nn.AvgPool1d(15, stride=1, padding=7, count_include_pad=False)

        self.tfencoderlayerseason = nn.TransformerEncoderLayer(seasonconvchannel, 8, dim_feedforward=1024, dropout=0.3)
        self.tfencoderseason = nn.TransformerEncoder(self.tfencoderlayerseason, 3)

        self.tfencoderlayermonth = nn.TransformerEncoderLayer((monthconvchannel+seasonconvchannel), 8, dim_feedforward=1024, dropout=0.3)
        self.tfencodermonth = nn.TransformerEncoder(self.tfencoderlayermonth, 3)

        self.tfencoderlayerweek = nn.TransformerEncoderLayer((8+weekconvchannel+monthconvchannel+seasonconvchannel), 8, dim_feedforward=1024, dropout=0.3)
        self.tfencoderweek = nn.TransformerEncoder(self.tfencoderlayerweek, 3)

    def forward(self, refer):
        dateweek = refer[:,0,:].unsqueeze(1)
        datemonth = self.datepollingmonth(dateweek)
        dateseason = self.datepoolingseason(dateweek)

        day_x = refer[:,1:7,:]
        day_x_2 = self.conv1dday(day_x)
        day_x_2 = self.BNday(day_x_2)
        day_x_2 = self.ReLUlayer(day_x_2)
        day_x = torch.cat([day_x, day_x_2], dim=1)

        week_x = self.conv1dweek(day_x)
        week_x = self.BNweek(week_x)
        week_x = self.ReLUlayer(week_x)

        month_x = self.conv1dmonth1(week_x)
        month_x = self.BNmonth1(month_x)
        month_x = self.ReLUlayer(month_x)
        month_x = self.conv1dmonth2(month_x)
        month_x = self.BNmonth2(month_x)
        month_x = self.ReLUlayer(month_x)

        season_x = self.conv1dseason1(month_x)
        season_x = self.BNseason1(season_x)
        season_x = self.ReLUlayer(season_x)
        season_x = self.conv1dseason2(season_x)
        season_x = self.BNseason2(season_x)
        season_x = self.ReLUlayer(season_x)

        s_x_tfencode = self.dateseasonencoding(season_x, dateseason)
        s_x_tfencode = self.transpose2transformer(s_x_tfencode)
        s_x_tfencode = self.tfencoderseason(s_x_tfencode)

        ms_x = self.interpseason(season_x)
        ms_x = torch.cat([month_x, ms_x], dim=1)

        ms_x_tfencode = self.datemonthencoding(ms_x, datemonth)
        ms_x_tfencode = self.transpose2transformer(ms_x_tfencode)
        ms_x_tfencode = self.tfencodermonth(ms_x_tfencode)

        wms_x = self.interpmonth(ms_x)
        wms_x = torch.cat([day_x, week_x, wms_x], dim=1)

        wms_x_tfencode = self.dateweekencoding(wms_x, dateweek)
        wms_x_tfencode = self.transpose2transformer(wms_x_tfencode)
        wms_x_tfencode = self.tfencoderweek(wms_x_tfencode)

        return wms_x_tfencode, ms_x_tfencode, s_x_tfencode

    def interpmonth(self, month_x):
        month_x = nn.functional.interpolate(month_x, size=ReferLen, mode='nearest')
        month_x = self.monthinterpavgpool(month_x)

        return month_x

    def interpseason(self, season_x):
        season_x = nn.functional.interpolate(season_x, size=ReferLen//5, mode='nearest')
        season_x = self.seasoninterpacgpool(season_x)

        return season_x

    def transpose2transformer(self, x):
        x = x.transpose(0, 2)
        x = x.transpose(1, 2)

        return x

# %%
class QueryEncoder(nn.Module):
    def __init__(self):
        super(QueryEncoder, self).__init__()

        self.ReLUlayer = nn.LeakyReLU()

        self.dateweekencoding = DateEncoding((8+weekconvchannel+monthconvchannel+seasonconvchannel), dropout=0.1, maxlen=QueryLen)

        self.conv1dday = nn.Conv1d(6, 2, 1, 1, padding=0, bias=True)
        self.BNday = nn.BatchNorm1d(2)

        self.conv1dweek = nn.Conv1d(8, weekconvchannel, 5, stride=1, padding=2, padding_mode='circular', bias=True)
        self.BNweek = nn.BatchNorm1d(weekconvchannel)

        self.datepollingmonth = nn.AvgPool1d(5, stride=5)
        self.datemonthencoding = DateEncoding((monthconvchannel+seasonconvchannel), dropout=0.1, maxlen=QueryLen//5)

        self.conv1dmonth1 = nn.Conv1d(weekconvchannel, monthconvchannel, 10, stride=5, padding=4, padding_mode='circular', bias=True)
        self.BNmonth1 = nn.BatchNorm1d(monthconvchannel)
        self.conv1dmonth2 = nn.Conv1d(monthconvchannel, monthconvchannel, 4, stride=1, padding='same', padding_mode='circular', bias=True)
        self.BNmonth2 = nn.BatchNorm1d(monthconvchannel)

        self.datepoolingseason = nn.AvgPool1d(20, stride=20)
        self.dateseasonencoding = DateEncoding(seasonconvchannel, dropout=0.1, maxlen=QueryLen//20)

        self.conv1dseason1 = nn.Conv1d(monthconvchannel, seasonconvchannel, 8, stride=4, padding=3, padding_mode='circular', bias=True)
        self.BNseason1 = nn.BatchNorm1d(seasonconvchannel)
        self.conv1dseason2 = nn.Conv1d(seasonconvchannel, seasonconvchannel, 3, stride=1, padding=1, padding_mode='circular', bias=True)
        self.BNseason2 = nn.BatchNorm1d(seasonconvchannel)

        self.seasoninterpacgpool = nn.AvgPool1d(9, stride=1, padding=4, count_include_pad=False)
        self.monthinterpavgpool = nn.AvgPool1d(15, stride=1, padding=7, count_include_pad=False)

    def forward(self, query):
        dateweek = query[:,0,:].unsqueeze(1)
        datemonth = self.datepollingmonth(dateweek)
        dateseason = self.datepoolingseason(dateweek)

        day_x = query[:,1:7,:]
        day_x_2 = self.conv1dday(day_x)
        day_x_2 = self.BNday(day_x_2)
        day_x_2 = self.ReLUlayer(day_x_2)
        day_x = torch.cat([day_x, day_x_2], dim=1)

        week_x = self.conv1dweek(day_x)
        week_x = self.BNweek(week_x)
        week_x = self.ReLUlayer(week_x)
        
        month_x = self.conv1dmonth1(week_x)
        month_x = self.BNmonth1(month_x)
        month_x = self.ReLUlayer(month_x)
        month_x = self.conv1dmonth2(month_x)
        month_x = self.BNmonth2(month_x)
        month_x = self.ReLUlayer(month_x)

        season_x = self.conv1dseason1(month_x)
        season_x = self.BNseason1(season_x)
        season_x = self.ReLUlayer(season_x)
        season_x = self.conv1dseason2(season_x)
        season_x = self.BNseason2(season_x)
        season_x = self.ReLUlayer(season_x)

        s_qe2tfdecoder = self.dateseasonencoding(season_x, dateseason)
        s_qe2tfdecoder = self.transpose2transformer(s_qe2tfdecoder)

        ms_x = self.interpseason(season_x)
        ms_x = torch.cat([month_x, ms_x], dim=1)

        ms_qe2tfdecoder = self.datemonthencoding(ms_x, datemonth)
        ms_qe2tfdecoder = self.transpose2transformer(ms_qe2tfdecoder)

        wms_x = self.interpmonth(ms_x)
        wms_x = torch.cat([day_x, week_x, wms_x], dim=1)

        wms_qe2tfdecoder = self.dateweekencoding(wms_x, dateweek)
        wms_qe2tfdecoder = self.transpose2transformer(wms_qe2tfdecoder)

        return wms_qe2tfdecoder, ms_qe2tfdecoder, s_qe2tfdecoder

    def interpmonth(self, month_x):
        month_x = nn.functional.interpolate(month_x, size=QueryLen, mode='nearest')
        month_x = self.monthinterpavgpool(month_x)

        return month_x

    def interpseason(self, season_x):
        season_x = nn.functional.interpolate(season_x, size=QueryLen//5, mode='nearest')
        season_x = self.seasoninterpacgpool(season_x)

        return season_x

    def transpose2transformer(self, x):
        x = x.transpose(0, 2)
        x = x.transpose(1, 2)

        return x


# %%
class SequenceConvTransformer(nn.Module):
    def __init__(self):
        super(SequenceConvTransformer, self).__init__()

        self.ReLUlayer = nn.LeakyReLU()

        self.referencoder = ReferEncoder()

        self.queryencoder = QueryEncoder()

        self.tfdecoderlayerseason = nn.TransformerDecoderLayer(seasonconvchannel, 8, dim_feedforward=1024, dropout=0.3)
        self.tfdecoderseason = nn.TransformerDecoder(self.tfdecoderlayerseason, 3)

        self.tfdecoderlayermonth = nn.TransformerDecoderLayer((monthconvchannel+seasonconvchannel), 8, dim_feedforward=1024, dropout=0.3)
        self.tfdecodermonth = nn.TransformerDecoder(self.tfdecoderlayermonth, 3)

        self.tfdecoderlayerweek = nn.TransformerDecoderLayer((8+weekconvchannel+monthconvchannel+seasonconvchannel), 8, dim_feedforward=1024, dropout=0.3)
        self.tfdecoderweek = nn.TransformerDecoder(self.tfdecoderlayerweek, 3)

        self.rdconv1dweek = nn.Conv1d((8+weekconvchannel+monthconvchannel+seasonconvchannel), rdconvchannel, 1, stride=1, bias=True)
        self.rdBNweek = nn.BatchNorm1d(rdconvchannel)

        self.rdconv1dmonth = nn.Conv1d((monthconvchannel+seasonconvchannel), rdconvchannel, 1, stride=1, bias=True)
        self.rdBNmonth = nn.BatchNorm1d(rdconvchannel)

        self.rdconv1dseason = nn.Conv1d(seasonconvchannel, rdconvchannel, 1, stride=1, bias=True)
        self.rdBNseason = nn.BatchNorm1d(rdconvchannel)

        self.flattenlayer = nn.Flatten()
        self.tanhlayer = nn.Tanh()

        self.linearout1 = nn.Linear(4800, 4)
        self.BNout1 = nn.BatchNorm1d(4)
        self.linearout2 = nn.Linear(4, 1)
        self.BNout2 = nn.BatchNorm1d(1)

    def forward(self, refer, query):
        refer_week_encode, refer_month_encode, refer_season_encode = self.referencoder(refer)

        query_week_target, query_month_target, query_season_target = self.queryencoder(query)

        tf_out_week = self.tfdecoderweek(query_week_target, refer_week_encode)
        tf_out_month = self.tfdecodermonth(query_month_target, refer_month_encode)
        tf_out_season = self.tfdecoderseason(query_season_target, refer_season_encode)

        tf_out_week = self.transpose2Conv(tf_out_week)
        tf_out_month = self.transpose2Conv(tf_out_month)
        tf_out_season = self.transpose2Conv(tf_out_season)

        out_week = self.rdconv1dweek(tf_out_week)
        out_week = self.rdBNweek(out_week)
        out_week = self.ReLUlayer(out_week)

        out_month = self.rdconv1dmonth(tf_out_month)
        out_month = self.rdBNmonth(out_month)
        out_month = self.ReLUlayer(out_month)

        out_season = self.rdconv1dseason(tf_out_season)
        out_season = self.rdBNseason(out_season)
        out_season = self.ReLUlayer(out_season)

        out_week = self.flattenlayer(out_week)
        out_month = self.flattenlayer(out_month)
        out_season = self.flattenlayer(out_season)

        out = torch.cat([out_week, out_month, out_season], dim=1)
        out = self.linearout1(out)
        out = self.BNout1(out)
        out = self.tanhlayer(out)
        out = self.linearout2(out)
        out = self.BNout2(out)
        out = self.tanhlayer(out)

        return out

    def transpose2Conv(self, x):
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)

        return x



