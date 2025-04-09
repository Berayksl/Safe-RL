import roboticstoolbox as rtb
import numpy as np
from simulator import Continuous2DEnv, UnicycleDynamics, ModifiedUnicycleDynamics



agent = rtb.mobile.Unicycle()
print(agent)

agent.step((1,1), True)

anim = agent.run_animation(T=20)