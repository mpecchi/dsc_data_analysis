# %%
from dsc_data_analysis.dsc import MeasurePint

c = MeasurePint("kg", name="test")
c.add(1000, "kg")
c.add(2, "t")
c.add(1100, "kg")
# print(c.stk())
print(c.ave())
print(c.std())

d = MeasurePint("kg")
d.add([1, 2], "kg")
d.add([3, 4], "kg")
d.add([5, 6], "kg")
print(d.ave())
print(d.std())

# %%
