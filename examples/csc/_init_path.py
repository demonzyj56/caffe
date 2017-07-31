import sys
import os
__this_dir = os.path.dirname(__file__)
__caffe_path = os.path.join(__this_dir, '..', '..', 'python')
if __caffe_path not in sys.path:
    sys.path.insert(0, __caffe_path)
