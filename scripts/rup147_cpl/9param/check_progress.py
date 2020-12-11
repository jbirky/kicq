import emcee
import os

#os.system('rsync -az jbirky@mox.hyak.uw.edu:/gscratch/astro/jbirky/projects/kicq /astro/users/jbirky/projects')
 
reader = emcee.backends.HDFBackend("results/Rup147_emcee_node1.h5", read_only=True)
chain = reader.get_chain()
print('node1', chain.shape)

reader = emcee.backends.HDFBackend("results/Rup147_emcee_node2.h5", read_only=True)
chain = reader.get_chain()
print('node2', chain.shape)
