from dataclasses import dataclass, field


@dataclass
class OSCARSessionHeader:
    magicnumber: int = 0
    version: int = 0
    filetype: int  = 0
    deviceid: int = 0
    sessionid: int = 0
    sfirst: int = 0
    slast: int = 0
    compmethod: int = 0
    machtype: int = 0
    datasize: int = 0
    crc16: int = 0


@dataclass
class OSCARSessionEvent:
    ts1: int = 0
    ts2: int = 0
    evcount: int = 0
    t8: int = 0
    rate: float = 0.0
    gain: float = 0.0
    offset: float = 0.0
    mn: float = 0.0
    mx: float = 0.0
    len_dim: int = 0
    dim: str = ""
    second_field: bool = False
    mn2: float = 0.0
    mx2: float = 0.0
    data: list[int] = field(default_factory=list[int])
    data2: list[int] = field(default_factory=list[int])
    time: list[int] = field(default_factory=list[int])

@dataclass
class OSCARSessionChannel:
    code: int = 0
    size2: int = 0
    events: list[OSCARSessionEvent] = field(default_factory=list[OSCARSessionEvent])


@dataclass
class OSCARSessionData:
    mcsize: int = 0
    channels: list[OSCARSessionChannel] = field(default_factory=list[OSCARSessionChannel])


@dataclass
class OSCARSession:
    header: OSCARSessionHeader = None
    data: OSCARSessionData = None
