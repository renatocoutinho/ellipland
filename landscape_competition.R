library(rPython)

# loads python library
python.load('landscape_nspecies_gen.py')
# loads landscape file
python.exec("l = loadtxt('landA.txt')")
# actually runs the solver
# this should allow manipulation using R variables, both the landscape and the
# parameters. Unfortunately, the rPython library doesn't (yet?) support arrays
# nor OrderedDict's (the type of "parn").
python.exec("s = solve_landscape_nspecies(l, parn, 0.1)")

# converts array output to list of lists of lists, and get it into an R array
python.exec('slist = s.tolist()')
python.exec('shap = s.shape')
shape <- python.get('shap')
# is there a better way?
s <- as.array(unlist(python.get('slist')))
# it looks like filling of lists <-> arrays is in reversed order
dim(s) <- rev(shape)

