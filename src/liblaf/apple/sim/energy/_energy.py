from liblaf.apple import struct
from liblaf.apple.sim import obj as _o


class Energy(_o.Object):
    @property
    def objects(self) -> struct.NodeCollection[_o.Object]:
        raise NotImplementedError
