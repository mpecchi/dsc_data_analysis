from pint import UnitRegistry

ur = UnitRegistry()
ur.formatter.default_format = "~P"
qt = ur.Quantity
