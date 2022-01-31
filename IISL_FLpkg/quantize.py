import random
from numpy.linalg import norm

def quantize(g, s):
  select_list = [0, 1]
  quan_g = g.copy()
  if(len(g.shape) == 1):
    g_abs = norm(g, 2)
    if(g_abs == 0):
      return g
    for i in range(len(g)):
      for l in range(s):
        if(g_abs * (l/s) <= abs(g[i]) and abs(g[i]) < g_abs * (l+1) / s):
          p = (abs(g[i]) / g_abs) * s - l
          distri = [1-p, p]
          l_temp = random.choices(select_list, distri)[0]
          quan_g[i] = (l + l_temp) / s * g_abs
          if(g[i] < 0):
            quan_g[i] = -1 * quan_g[i]
          break
    return quan_g
  elif(len(g.shape) == 2):
    for i in range(len(g)):
      for j in range(len(g[i])):
        g_abs = norm(g[i], 2)
        if(g_abs != 0):
          for l in range(s):
            if(g_abs * (l/s) <= abs(g[i][j]) and abs(g[i][j]) < g_abs * (l+1) / s):
              p = (abs(g[i][j]) / g_abs) * s - l
              distri = [1-p, p]
              l_temp = random.choices(select_list, distri)[0]
              quan_g[i][j] = (l + l_temp) / s * g_abs
              if(g[i][j] < 0):
                quan_g[i][j] = -1 * quan_g[i][j]
              break
    return quan_g
  elif(len(g.shape) == 4):   
    for i in range(len(g)):
      for j in range(len(g[i])):
        for k in range(len(g[i][j])):
          for l in range(len(g[i][j][k])):
            g_abs = norm(g[i][j][k], 2)
            if(g_abs != 0):
              for l in range(s):
                if(g_abs * (l/s) <= abs(g[i][j][k][l]) and abs(g[i][j][k][l]) < g_abs * (l+1) / s):
                  p = (abs(g[i][j][k][l]) / g_abs) * s - l
                  distri = [1-p, p]
                  l_temp = random.choices(select_list, distri)[0]
                  quan_g[i][j][k][l] = (l + l_temp) / s * g_abs
                  if(g[i][j][k][l] < 0):
                    quan_g[i][j][k][l] = -1 * quan_g[i][j][k][l]
                  break
    return quan_g 