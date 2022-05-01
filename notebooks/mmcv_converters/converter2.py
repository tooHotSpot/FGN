from .builder2 import CONVERTERS_BUILDER


# use the registry to manage the module
@CONVERTERS_BUILDER.register_module()
class Converter2(object):
    def __init__(self, c, d):
        self.new_c = c
        self.new_d = d
