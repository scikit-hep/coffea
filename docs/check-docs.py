from coffea.nanoevents.methods.vector import LorentzVector as lv
# looping over members defined in this class and no base classes
for name in lv.__dict__.keys():
    # skip hidden stuff
    if name.startswith('_'):
        continue
    member = getattr(lv, name)
    # filter out un-documented
    if getattr(member, '__doc__') is None:
        continue
    print(name, type(member))
    print(member.__doc__)
