from enum import Enum, auto


# Groups
GRP_CPAP = "CPAP"
GRP_POS = "POS"
GRP_OXI = "OXI"
GRP_JOURNAL = "JOURNAL"
GRP_SLEEP = "SLEEP"

# Internal graph identifiers -- must NOT be translated

STR_GRAPH_EventBreakdown = "EventBreakdown"
STR_GRAPH_SleepFlags = "SF"
STR_GRAPH_FlowRate = "FlowRate"
STR_GRAPH_Pressure = "Pressure"
STR_GRAPH_LeakRate = "Leak"
STR_GRAPH_FlowLimitation = "FLG"
STR_GRAPH_Snore = "Snore"
STR_GRAPH_TidalVolume = "TidalVolume"
STR_GRAPH_MaskPressure = "MaskPressure"
STR_GRAPH_RespRate = "RespRate"
STR_GRAPH_MinuteVent = "MinuteVent"
STR_GRAPH_PTB = "PTB"
STR_GRAPH_RespEvent = "RespEvent"
STR_GRAPH_Ti = "Ti"
STR_GRAPH_Te = "Te"
STR_GRAPH_SleepStage = "SleepStage"
STR_GRAPH_Inclination = "Inclination"
STR_GRAPH_Orientation = "Orientation"
STR_GRAPH_Motion = "Motion"
STR_GRAPH_TestChan1 = "TestChan1"
STR_GRAPH_TestChan2 = "TestChan2"
STR_GRAPH_AHI = "AHI"
STR_GRAPH_Weight = "Weight"
STR_GRAPH_BMI = "BMI"
STR_GRAPH_Zombie = "Zombie"
STR_GRAPH_Sessions = "Sessions"
STR_GRAPH_SessionTimes = "SessionTimes"
STR_GRAPH_Usage = "Usage"
STR_GRAPH_PeakAHI = "PeakAHI"
STR_GRAPH_TAP = "TimeAtPressure"
STR_GRAPH_Oxi_Pulse = "Pulse"
STR_GRAPH_Oxi_SPO2 = "SPO2"
STR_GRAPH_Oxi_Plethy = "Plethy"
STR_GRAPH_Oxi_Perf = "Perf. Index"
STR_GRAPH_Oxi_PulseChange = "PulseChange"
STR_GRAPH_Oxi_SPO2Drop = "SPO2Drop"
STR_GRAPH_ObstructLevel = "ObstructLevel"
STR_GRAPH_PressureMeasured = "PressureMeasured"
STR_GRAPH_rRMV = "rRMV"
STR_GRAPH_rMVFluctuation = "rMVFluctuation"
STR_GRAPH_FlowFull = "FlowFull"
STR_GRAPH_SPRStatus = "SPRStatus"


#Units
STR_UNIT_M = " m"
STR_UNIT_CM = " cm"
STR_UNIT_INCH = "in"
STR_UNIT_FOOT = "ft"
STR_UNIT_POUND = "lb"
STR_UNIT_OUNCE = "oz"
STR_UNIT_KG = "kg"
STR_UNIT_CMH2O = "cmH2O"
STR_UNIT_Hours = "Hours"
STR_UNIT_Minutes = "Minutes"
STR_UNIT_Seconds = "Seconds"
STR_UNIT_h = "h" # hours shortform
STR_UNIT_m = "m" # minutes shortform
STR_UNIT_s = "s" # seconds shortform
STR_UNIT_ms = "ms" # milliseconds
STR_UNIT_EventsPerHour = "Events/hr" # Events per hour
STR_UNIT_Percentage = "%"
STR_UNIT_Hz = "Hz"          # Hertz
STR_UNIT_BPM = "bpm"        # Beats per Minute
STR_UNIT_LPM = "l/min"      # Litres per Minute
STR_UNIT_Litres = "Litres"
STR_UNIT_ml = "ml"        # millilitres
STR_UNIT_BreathsPerMinute = "Breaths/min" # Breaths per minute
STR_UNIT_Unknown = "?"
STR_UNIT_Ratio = "ratio"
STR_UNIT_Severity = "Severity (0-1)"
STR_UNIT_Degrees = "Degrees"

# Machine type
class MachineType(Enum):
    T_UNKNOWN = 0
    MT_CPAP = auto()
    MT_OXIMETER = auto()
    MT_SLEEPSTAGE = auto()
    MT_JOURNAL = auto()
    MT_POSITION = auto()
    MT_UNCATEGORIZED = 99

# ScopeType
class ScopeType(Enum):
    GLOBAL = 0
    MACHINE = auto()
    DAY = auto()
    SESSION = auto()

# ChannelType
class ChanType(Enum):
    DATA = 1,
    SETTING = 2,
    FLAG = 4,
    MINOR_FLAG = 8,
    SPAN = 16,
    WAVEFORM = 32,
    UNKNOWN = 64


# ChannelID
class ChannelID(Enum):
    CPAP_Pressure= 0x0110C
    CPAP_IPAP = 0x110D
    CPAP_IPAPLo = 0x1110
    CPAP_IPAPHi = 0x1111
    CPAP_EPAP = 0x110E
    CPAP_EPAPLo = 0x111C
    CPAP_EPAPHi= 0x111D
    CPAP_PS = 0x110F
    CPAP_Mode  = 0x1200
    CPAP_AHI = 0x1116
    CPAP_PressureMin = 0x1020
    CPAP_PressureMax = 0x1021
    CPAP_Ramp = 0x1027
    CPAP_RampTime= 0x1022
    CPAP_RampPressure = 0x1023
    CPAP_Obstructive = 0x1002
    CPAP_Hypopnea = 0x1003
    CPAP_AllApnea = 0x1010
    CPAP_ClearAirway = 0x1001
    CPAP_Apnea = 0x1004
    CPAP_PB = 0x1028
    CPAP_CSR = 0x1000
    CPAP_LeakFlag = 0x100a
    CPAP_ExP = 0x100c
    CPAP_NRI = 0x100b
    CPAP_VSnore = 0x1007
    CPAP_VSnore2 = 0x1008
    CPAP_RERA = 0x1006
    CPAP_FlowLimit = 0x1005
    CPAP_SensAwake = 0x100d
    CPAP_FlowRate = 0x1100
    CPAP_MaskPressure = 0x1101
    CPAP_MaskPressureHi  = 0x1102
    CPAP_RespEvent = 0x1112
    CPAP_Snore  = 0x1104
    CPAP_MinuteVent = 0x1105
    CPAP_RespRate  = 0x1106
    CPAP_TidalVolume = 0x1103
    CPAP_PTB = 0x1107
    CPAP_LargeLeak= 0x1158
    CPAP_Leak = 0x1108
    CPAP_LeakMedian = 0x1118
    CPAP_LeakTotal = 0x1117
    CPAP_MaxLeak = 0x1115
    CPAP_FLG = 0x1113
    CPAP_IE = 0x1109
    CPAP_Te = 0x110A
    CPAP_Ti = 0x110B
    CPAP_TgMV = 0x1114
    CPAP_UserFlag1 = 0x101e
    CPAP_UserFlag2 = 0x101f
    CPAP_UserFlag3 = 0x1024
    CPAP_RDI = 0x1119
    CPAP_PSMin = 0x111A
    CPAP_PSMax = 0x111B
    CPAP_PressureSet = 0x11A4
    CPAP_IPAPSet = 0x11A5
    CPAP_EPAPSet = 0x11A6
    CPAP_EEPAP = 0x11A7
    OXI_Pulse  = 0x1800
    OXI_SPO2 = 0x1801
    OXI_Perf = 0x1805
    OXI_PulseChange = 0x1803
    OXI_SPO2Drop = 0x1804
    OXI_Plethy  = 0x1802
    POS_Orientation = 0x2990
    POS_Inclination = 0x2991
    POS_Movement = 0x2992
    RMS9_MaskOnTime = 0x1025
    CPAP_SummaryOnly  = 0x1026

#            Group     ChannelID                Code               Type                 Scope              Lookup       FieldType       Color
CHANNELS = [[GRP_CPAP, ChannelID.CPAP_Pressure, ChanType.WAVEFORM, MachineType.MT_CPAP, ScopeType.SESSION, 'Pressure',  STR_UNIT_CMH2O, "red"],
            [GRP_CPAP, ChannelID.CPAP_IPAP,     ChanType.WAVEFORM, MachineType.MT_CPAP, ScopeType.SESSION, "IPAP",      STR_UNIT_CMH2O, "red"],
            [GRP_CPAP, ChannelID.CPAP_IPAPLo,   ChanType.WAVEFORM, MachineType.MT_CPAP, ScopeType.SESSION, "IPAPLo",    STR_UNIT_CMH2O, "orange"],
            [GRP_CPAP, ChannelID.CPAP_IPAPHi,   ChanType.WAVEFORM, MachineType.MT_CPAP, ScopeType.SESSION, "IPAPHi",    STR_UNIT_CMH2O,   "orange"],
            [GRP_CPAP, ChannelID.CPAP_EPAP,     ChanType.WAVEFORM, MachineType.MT_CPAP, ScopeType.SESSION, "EPAP",       STR_UNIT_CMH2O,   "green"],
            [GRP_CPAP, ChannelID.CPAP_EPAPLo,   ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "EPAPLo",  STR_UNIT_CMH2O,            "light blue"],
            [GRP_CPAP, ChannelID.CPAP_EPAPHi,   ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "EPAPHi",  STR_UNIT_CMH2O,            "aqua"],
            [GRP_CPAP, ChannelID.CPAP_EEPAP,    ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "EEPAP",   STR_UNIT_CMH2O,            "purple"],
            [GRP_CPAP, ChannelID.CPAP_PS,       ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "PS",  STR_UNIT_CMH2O,            "grey"],
            [GRP_CPAP, ChannelID.CPAP_PSMin ,   ChanType.SETTING,     MachineType.MT_CPAP, ScopeType.SESSION, "PSMin",     STR_UNIT_CMH2O,            "dark cyan"],
            [GRP_CPAP, ChannelID.CPAP_PSMax,    ChanType.SETTING,     MachineType.MT_CPAP, ScopeType.SESSION, "PSMax",     STR_UNIT_CMH2O,            "dark magenta"],
            [GRP_CPAP, ChannelID.CPAP_PressureMin, ChanType.SETTING,     MachineType.MT_CPAP, ScopeType.SESSION, "PressureMin",  STR_UNIT_CMH2O,            "orange"],
            [GRP_CPAP, ChannelID.CPAP_PressureMax, ChanType.SETTING,     MachineType.MT_CPAP, ScopeType.SESSION, "PressureMax", STR_UNIT_CMH2O,            "light blue"],
            [GRP_CPAP, ChannelID.CPAP_RampTime, ChanType.SETTING,     MachineType.MT_CPAP, ScopeType.SESSION, "RampTime",       STR_UNIT_Minutes,          "black"],
            [GRP_CPAP, ChannelID.CPAP_RampPressure, ChanType.SETTING,     MachineType.MT_CPAP, ScopeType.SESSION, "RampPressure", STR_UNIT_CMH2O,         "black"],
            [GRP_CPAP, ChannelID.CPAP_Ramp,         ChanType.SPAN,        MachineType.MT_CPAP, ScopeType.SESSION, "Ramp",            STR_UNIT_EventsPerHour,    "light blue"],
            [GRP_CPAP, ChannelID.CPAP_PressureSet, ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "PressureSet",   STR_UNIT_CMH2O,            "dark red"],
            [GRP_CPAP, ChannelID.CPAP_IPAPSet, ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "IPAPSet",          STR_UNIT_CMH2O,            "dark red"],
            [GRP_CPAP, ChannelID.CPAP_EPAPSet, ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, "EPAPSet",       STR_UNIT_CMH2O,            "dark green"],
            [GRP_CPAP, ChannelID.CPAP_CSR, ChanType.SPAN,        MachineType.MT_CPAP, ScopeType.SESSION, "CSR",            STR_UNIT_Percentage,    "red"],
            [GRP_CPAP, ChannelID.CPAP_PB, ChanType.SPAN,        MachineType.MT_CPAP, ScopeType.SESSION, "PB",      STR_UNIT_Percentage,      "red"],
            [GRP_CPAP, ChannelID.CPAP_ClearAirway   , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "ClearAirway",        STR_UNIT_EventsPerHour,        "purple"],
            [GRP_CPAP, ChannelID.CPAP_Obstructive   , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "Obstructive", STR_UNIT_EventsPerHour,        "#40c0ff"],
            [GRP_CPAP, ChannelID.CPAP_Hypopnea      , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "Hypopnea",          STR_UNIT_EventsPerHour,        "blue"],
            [GRP_CPAP, ChannelID.CPAP_Apnea         , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "Apnea",            STR_UNIT_EventsPerHour,        "dark green"],
            [GRP_CPAP, ChannelID.CPAP_AllApnea      , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "AllApnea",          STR_UNIT_EventsPerHour,        "#40c0ff"],
            [GRP_CPAP, ChannelID.CPAP_FlowLimit     , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "FlowLimit",      STR_UNIT_EventsPerHour,        "#404040"],
            [GRP_CPAP, ChannelID.CPAP_RERA          , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "RERA",              STR_UNIT_EventsPerHour,        "red"],
            [GRP_CPAP, ChannelID.CPAP_VSnore        , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "VSnore",              STR_UNIT_EventsPerHour,        "red"],
            [GRP_CPAP, ChannelID.CPAP_VSnore2       , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "VSnore2",          STR_UNIT_EventsPerHour,        "red"],
            [GRP_CPAP, ChannelID.CPAP_LeakFlag      , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "LeakFlag",           STR_UNIT_EventsPerHour,        "light gray"],
            [GRP_CPAP, ChannelID.CPAP_LargeLeak     , ChanType.SPAN,        MachineType.MT_CPAP, ScopeType.SESSION, "LeakSpan",           STR_UNIT_EventsPerHour,        "light gray"],
            [GRP_CPAP, ChannelID.CPAP_NRI           , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "NRI",                STR_UNIT_EventsPerHour,        "orange"],
            [GRP_CPAP, ChannelID.CPAP_ExP           , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "ExP",             STR_UNIT_EventsPerHour,        "dark magenta"],
            [GRP_CPAP, ChannelID.CPAP_SensAwake     , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "SensAwake",        STR_UNIT_EventsPerHour,        "red"],
            [GRP_CPAP, ChannelID.CPAP_UserFlag1     , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "UserFlag1",        STR_UNIT_EventsPerHour,        "red"],
            [GRP_CPAP, ChannelID.CPAP_UserFlag2     , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "UserFlag2",         STR_UNIT_EventsPerHour,        "red"],
            [GRP_CPAP, ChannelID.CPAP_UserFlag3     , ChanType.FLAG,        MachineType.MT_CPAP, ScopeType.SESSION, "UserFlag3",      STR_UNIT_EventsPerHour,        "dark grey"],
            [GRP_OXI,  ChannelID.OXI_Pulse          , ChanType.WAVEFORM,    MachineType.MT_OXIMETER, ScopeType.SESSION, STR_GRAPH_Oxi_Pulse,    STR_UNIT_BPM,         "red"],
            [GRP_OXI,  ChannelID.OXI_SPO2           , ChanType.WAVEFORM,    MachineType.MT_OXIMETER, ScopeType.SESSION, STR_GRAPH_Oxi_SPO2,      STR_UNIT_Percentage,              "blue"],
            [GRP_OXI,  ChannelID.OXI_Plethy         , ChanType.WAVEFORM,    MachineType.MT_OXIMETER, ScopeType.SESSION, STR_GRAPH_Oxi_Plethy,       STR_UNIT_Hz,               "#404040"],
            [GRP_OXI,  ChannelID.OXI_Perf             , ChanType.WAVEFORM,   MachineType.MT_OXIMETER, ScopeType.SESSION, STR_GRAPH_Oxi_Perf,      STR_UNIT_Percentage,        "magenta"],
            [GRP_OXI,  ChannelID.OXI_PulseChange     , ChanType.FLAG,        MachineType.MT_OXIMETER, ScopeType.SESSION, STR_GRAPH_Oxi_PulseChange,               STR_UNIT_EventsPerHour,       "light grey"],
            [GRP_OXI,  ChannelID.OXI_SPO2Drop        , ChanType.FLAG,        MachineType.MT_OXIMETER, ScopeType.SESSION, STR_GRAPH_Oxi_SPO2Drop,             STR_UNIT_EventsPerHour,        "light blue"],
            [GRP_CPAP, ChannelID.CPAP_FlowRate          , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, STR_GRAPH_FlowRate, STR_UNIT_LPM,        "black"],
            [GRP_CPAP, ChannelID.CPAP_MaskPressure      , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, STR_GRAPH_MaskPressure, STR_UNIT_CMH2O,        "blue"],
            [GRP_CPAP, ChannelID.CPAP_MaskPressureHi   , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION,  "MaskPressureHi", STR_UNIT_CMH2O,        "blue"],
            [GRP_CPAP, ChannelID.CPAP_TidalVolume       , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, STR_GRAPH_TidalVolume, STR_UNIT_ml,        "magenta"],
            [GRP_CPAP, ChannelID.CPAP_Snore             , ChanType.WAVEFORM,    MachineType.MT_CPAP,  ScopeType.SESSION,STR_GRAPH_Snore, STR_UNIT_Unknown,           "grey"],
            [GRP_CPAP, ChannelID.CPAP_MinuteVent        , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, STR_GRAPH_MinuteVent, STR_UNIT_LPM,        "dark cyan"],
            [GRP_CPAP, ChannelID.CPAP_RespRate         , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION,  STR_GRAPH_RespRate, STR_UNIT_BreathsPerMinute,          "dark magenta"],
            [GRP_CPAP, ChannelID.CPAP_PTB               , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, STR_GRAPH_PTB, STR_UNIT_Percentage,       "dark grey"],
            [GRP_CPAP, ChannelID.CPAP_Leak              , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION, STR_GRAPH_LeakRate, STR_UNIT_LPM,        "dark green"],
            [GRP_CPAP, ChannelID.CPAP_IE                , ChanType.WAVEFORM,    MachineType.MT_CPAP,  ScopeType.SESSION,  "IE", STR_UNIT_Ratio,        "dark red"],
            [GRP_CPAP, ChannelID.CPAP_Te                , ChanType.WAVEFORM,    MachineType.MT_CPAP,  ScopeType.SESSION, STR_GRAPH_Te,         STR_UNIT_Seconds,      "dark green"],
            [GRP_CPAP, ChannelID.CPAP_Ti                , ChanType.WAVEFORM,    MachineType.MT_CPAP, ScopeType.SESSION,  STR_GRAPH_Ti,       STR_UNIT_Seconds,      "dark blue"],
            [GRP_CPAP, ChannelID.CPAP_RespEvent         , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION, STR_GRAPH_RespEvent, STR_UNIT_CMH2O,       "black"],
            [GRP_CPAP, ChannelID.CPAP_FLG               , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION, STR_GRAPH_FlowLimitation, STR_UNIT_Severity,          "#585858"],
            [GRP_CPAP, ChannelID.CPAP_TgMV              , ChanType.WAVEFORM,  MachineType.MT_CPAP,   ScopeType.SESSION, "TgMV", STR_UNIT_LPM,           "dark red"],
            [GRP_CPAP, ChannelID.CPAP_MaxLeak           , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION, "MaxLeak", STR_UNIT_LPM,        "dark red"],
            [GRP_CPAP, ChannelID.CPAP_AHI               , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION,  "AHI", STR_UNIT_EventsPerHour,   "dark red"],
            [GRP_CPAP, ChannelID.CPAP_LeakTotal         , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION,   "LeakTotal", STR_UNIT_LPM,        "dark green"],
            [GRP_CPAP, ChannelID.CPAP_LeakMedian        , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION,  "LeakMedian", STR_UNIT_LPM,        "dark green"],
            [GRP_CPAP, ChannelID.CPAP_RDI               , ChanType.WAVEFORM,   MachineType.MT_CPAP,  ScopeType.SESSION,  "RDI", STR_UNIT_EventsPerHour,   "dark red"],
            [GRP_POS, ChannelID.POS_Orientation        , ChanType.WAVEFORM,   MachineType.MT_POSITION,  ScopeType.SESSION, STR_GRAPH_Orientation,   STR_UNIT_Degrees,   "dark blue"],
            [GRP_POS, ChannelID.POS_Movement            , ChanType.WAVEFORM,  MachineType.MT_POSITION, ScopeType.SESSION, STR_GRAPH_Motion,   STR_UNIT_Unknown,   "dark green"],
            [GRP_CPAP, ChannelID.RMS9_MaskOnTime        , ChanType.DATA,   MachineType.MT_CPAP,  ScopeType.SESSION, "MaskOnTime",  STR_UNIT_Unknown,   "red"],
            [GRP_CPAP, ChannelID.CPAP_SummaryOnly      , ChanType.DATA,   MachineType.MT_CPAP,  ScopeType.SESSION, "SummaryOnly",   STR_UNIT_Unknown,   "red"],
            [GRP_CPAP, ChannelID.CPAP_Mode, ChanType.SETTING,   MachineType.MT_CPAP,  ScopeType.SESSION, "PAPMode", "red"]]