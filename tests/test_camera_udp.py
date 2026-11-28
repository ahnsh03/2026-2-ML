import struct,unittest
from quality_estimation.camera_udp import CameraAssembler,jpeg_size

JPEG=b'\xff\xd8\xff\xc0'+struct.pack('>H',17)+b'\x08'+struct.pack('>HH',720,1280)+b'\x03'+bytes(9)+b'\xff\xd9'
def packet(index,data,*,stamp=(123,500),final=True):
    return b'MOR'+struct.pack('<IIii',*stamp,index,len(data))+data+b'padding'+(b'EI'if final else b'AI')
class CameraUDPTest(unittest.TestCase):
    def test_native_padding_is_not_image_data(self):
        a=CameraAssembler();out=a.feed(packet(0,JPEG),0);assert out[0]==JPEG;assert(out[1]['width'],out[1]['height'])==(1280,720)
    def test_fragment_gap_never_silently_reassembles(self):
        a=CameraAssembler();assert a.feed(packet(0,JPEG[:8],final=False),0)is None
        assert a.feed(packet(2,JPEG[8:]),.1)is None
        assert a.count['sequence_or_stamp_error']==1
        assert a.feed(packet(0,JPEG),.2)[0]==JPEG
    def test_cross_stamp_or_timeout_discards_partial(self):
        a=CameraAssembler();a.feed(packet(0,JPEG[:8],final=False),0)
        assert a.feed(packet(1,JPEG[8:],stamp=(124,500)),.1)is None
        a.feed(packet(0,JPEG[:8],final=False),1)
        assert a.feed(packet(1,JPEG[8:]),2)is None;assert a.count['assembly_timeout']==1
    def test_contiguous_fragments_preserve_source(self):
        a=CameraAssembler();a.feed(packet(0,JPEG[:8],final=False),0);data,meta=a.feed(packet(1,JPEG[8:]),.1)
        assert data==JPEG;assert meta['fragments']==2
    def test_malformed_and_oversized_rejected(self):
        a=CameraAssembler(max_bytes=5);assert a.feed(packet(0,JPEG),0)is None;assert a.count['oversize_frame']==1
        a=CameraAssembler();assert a.feed(b'MOR',0)is None;assert a.count['invalid_packet']==1
        assert a.feed(packet(0,b'notajpeg'),1)is None;assert a.count['invalid_jpeg']==1
