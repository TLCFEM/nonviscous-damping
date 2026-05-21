model = """node 1 0 0
node 2 1 0

material Bilinear1D 1 100 1 .01 0

element T2D2 1 1 2 1 1
element Mass 2 2 1 1

fix2 1 1 1
fix2 2 2 1 2

hdf5recorder 1 Node U 2
hdf5recorder 2 Node RF 2

amplitude Sine 1 {period} 0.02

displacement 1 1 1 1 2

step dynamic 1 2
set ini_step_size 1E-2
set fixed_step_size 1

integrator UDDNewmark 1 .25 .5 -2 0 10 0

converger AbsIncreDisp 2 1E-14 10 0

analyze

peek node 2

save recorder 1 2

exit
"""
