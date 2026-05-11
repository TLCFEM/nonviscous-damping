# EXAMPLE 1

node 1 0 0
node 2 1 0

material Elastic1D 1 100

element T2D2 1 1 2 1 1
element Mass 2 2 1 1

fix2 1 1 1
fix2 2 2 1 2

hdf5recorder 1 Node U 2

initial displacement 1 1 2
initial acceleration -100 1 2

step dynamic 1 10
set ini_step_size 1E-2
set fixed_step_size 1

integrator UDDNewmark 1 .25 .5 -2 0 10 0
# integrator UDANewmark 1 .25 .5 2 0 10 0

converger AbsIncreDisp 2 1E-14 10 1

analyze

save recorder 1

exit