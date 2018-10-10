import matplotlib.pyplot as plt
def GridMaker(L,p):
    import numpy as np
    Grid = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            rand = np.random.random()
            if rand < p:
                Grid[i,j] = 1
            else:
                Grid[i,j] = 0
    return Grid
def Clusterfinder(Grid):
    L = Grid.GetL()
    counter = 2
    for i in range(L):
        for j in range(L):
            P = Grid.GetPoint(i,j)
            if  P > 1:
                #Already a cluster
                pass
            elif P == 1:
                #conducting not in cluster
                Grid.ClusterExplorer(i,j,counter)
                counter += 1
class Grid:
    def __init__(self,L,p):
        self.L = L
        self.Beads = GridMaker(L,p)
    def Plot(self):
        plt.imshow(self.Beads)
        plt.show()
    def GetL(self):
        return self.L
    def GetPoint(self,i,j):
        return self.Beads[i,j]
    def ChangePoint(self,i,j,val):
        self.Beads[i,j] = val
    def ClusterExplorer(self,i,j,cnum,Taken):
        Taken.append((i,j))
        if self.GetPoint(i,j) != 1:
            raise ValueError("Point not conducting!!")
        print(Taken)
        print("looking at ",i,j)
        #Rename current point
        self.ChangePoint(i,j,cnum)
        #Explore neighboring points
        i_s = [1,-1,0,0]
        j_s = [0,0,1,-1]
        for a,b in zip(i_s,j_s):
            print("Exploring ",i+a,j+b)
            if (i+a,j+b) in Taken:
                print("Taken")
                #Already found in cluster
                continue
            if i+a > self.L or i+a < 0 or j+b > self.L or j+b < 0:
                print("Outside")
                #outside grid, pass
                continue
            P = self.GetPoint(i+a,j+b)
            if P > 1:
                print("Other cluster")
                #A different cluster
                continue
            if P == 1:
                print("Yay point in new cluster")
                #Belongs to this cluster, clusterfind it
                T2 = self.ClusterExplorer(i+a,j+b,cnum,Taken)
                for e in T2:
                    Taken.append(e)
                continue
        print("Returning")
        return Taken
a = Grid(10,0.5)
a.Plot()
a.ClusterExplorer(0,0,2,[])
a.Plot()
