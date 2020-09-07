from data_types_module import dword, word, char
from utils import color
from obj import Obj
from collections import namedtuple
from math import sin, cos, tan


BLACK = color(0, 0, 0)
WHITE = color(1, 1, 1)
RED = color(1, 0, 0)
PI = 3.141592653589793

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z','w'])


def sum(v0, v1):
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def subV2(v0,v1):
	return V2(v0.x -v1.x, v0.y - v1.y)

def mul(v0, k):
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
	
def length(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5


def norm(v0):
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)


def bbox(*vertices):
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]

    xs.sort()
    ys.sort()

    xmin = xs[0]
    xmax = xs[-1]
    ymin = ys[0]
    ymax = ys[-1]

    return xmin, xmax, ymin, ymax

def cross(v1, v2):
    return V3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
    )

def baryCoords(A, B, C, P):
    # u es para la A, v es para B, w para C
    try:
        u = ( ((B.y - C.y)*(P.x - C.x) + (C.x - B.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        v = ( ((C.y - A.y)*(P.x - C.x) + (A.x - C.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w

def _deg2rad(degrees):
	radians = degrees * (PI/180)
	return radians

#only for square matrix
def matrixMul(matrix1, matrix2, isVector=False):
	matrix = [[0 for x in range(len(matrix1))] for y in range(len(matrix2[0]))]
	for i in range(len(matrix1)):
		for j in range(len(matrix2[0])):
			for k in range(len(matrix2)):
				matrix[i][j] += matrix1[i][k] * matrix2[k][j]
	return matrix

def matXvect(matrix1, vector):
	matrix = [[0 for x in range(len(matrix1))] for y in range(1)]
	for i in range(len(matrix1)):
		for j in range(1):
			for k in range(len(vector)):
				matrix[0][i] += matrix1[i][k] * vector[k]
	return matrix


def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                print("MATRIX NOT INVERTIBLE")
                return -1
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret

class Raytracer(object):

	#constructor
	def __init__(self, width, height):
		self.framebuffer = []
		self.curr_color = WHITE
		self.glCreateWindow(width, height)
		
		self.camPosition = V3(0,0,0)
		self.fov = 60

		self.scene = []


	def glCreateWindow(self, width, height):
		#width and height for the framebuffer
		self.width = width
		self.height = height
		self.glViewport(0,0,width,height)
		self.glClear()

	def glInit(self):
		self.curr_color = BLACK

	def glViewport(self, x, y, width, height):
		self.viewportX = x
		self.viewportY = y
		self.viewportWidth = width
		self.viewportHeight = height

		self.viewportMatrix = 	[[width/2,0,0,x + width/2],
								[0,height/2,0,y +height/2],
								[0,0,0.5,0.5],
								[0,0,0,1]]

	def glClear(self):
		self.framebuffer = [[BLACK for x in range(
		    self.width)] for y in range(self.height)]
		
		#Zbuffer (buffer de profundidad)
		self.zbuffer = [ [ float('inf') for x in range(self.width)] for y in range(self.height) ]
		

	def glClearColor(self, r, g, b):
		clearColor = color(
				round(r * 255),
				round(g * 255),
				round(b * 255)
			)

		self.framebuffer = [[clearColor for x in range(
		    self.width)] for y in range(self.height)]

	def glVertex(self, x, y):
		#las funciones fueron obtenidas de https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glViewport.xml
		X = round((x+1) * (self.viewportWidth/2) + self.viewportX)
		Y = round((y+1) * (self.viewportHeight/2) + self.viewportY)
		self.point(X, Y)
		

	def glVertex_coord(self, x, y, color= None):

		if x < self.viewportX or x >= self.viewportX + self.viewportWidth or  y < self.viewportY or y >= self.viewportY + self.viewportHeight:
			return 

		if x >= self.width or x < 0 or y >= self.height or y < 0:
			return
		try:
			self.framebuffer[y][x] = color or self.curr_color
		except:
			pass

	def glColor(self, r, g, b):
		self.curr_color = color(round(r / 255), round(g / 255), round(b / 255))

	def point(self, x, y):
		self.framebuffer[x][y] = self.curr_color

	def glFinish(self, filename):
		archivo = open(filename, 'wb')

		# File header 14 bytes
		archivo.write(char("B"))
		archivo.write(char("M"))
		archivo.write(dword(14+40+self.width*self.height))
		archivo.write(dword(0))
		archivo.write(dword(14+40))

		#Image Header 40 bytes
		archivo.write(dword(40))
		archivo.write(dword(self.width))
		archivo.write(dword(self.height))
		archivo.write(word(1))
		archivo.write(word(24))
		archivo.write(dword(0))
		archivo.write(dword(self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))

		#Pixeles, 3 bytes cada uno

		for x in range(self.height):
			for y in range(self.width):
				archivo.write(self.framebuffer[x][y])

		#Close file
		archivo.close()

		
	def glZBuffer(self, filename):
		archivo = open(filename, 'wb')

		# File header 14 bytes
		archivo.write(bytes('B'.encode('ascii')))
		archivo.write(bytes('M'.encode('ascii')))
		archivo.write(dword(14 + 40 + self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(14 + 40))

		# Image Header 40 bytes
		archivo.write(dword(40))
		archivo.write(dword(self.width))
		archivo.write(dword(self.height))
		archivo.write(word(1))
		archivo.write(word(24))
		archivo.write(dword(0))
		archivo.write(dword(self.width * self.height * 3))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))
		archivo.write(dword(0))

		# Minimo y el maximo
		minZ = float('inf')
		maxZ = -float('inf')
		for x in range(self.height):
			for y in range(self.width):
				if self.zbuffer[x][y] != -float('inf'):
					if self.zbuffer[x][y] < minZ:
						minZ = self.zbuffer[x][y]

					if self.zbuffer[x][y] > maxZ:
						maxZ = self.zbuffer[x][y]

		for x in range(self.height):
			for y in range(self.width):
				depth = self.zbuffer[x][y]
				if depth == -float('inf'):
					depth = minZ
				depth = (depth - minZ) / (maxZ - minZ)
				archivo.write(color(depth,depth,depth))

		archivo.close()

	def rtRender(self):
		#pixel por pixel
		for y in range(self.height):
			for x in range(self.width):

				# pasar valor de pixel a coordenadas NDC (-1 a 1)
				Px = 2 * ( (x+0.5) / self.width) - 1
				Py = 2 * ( (y+0.5) / self.height) - 1

				#FOV(angulo de vision), asumiendo que el near plane esta a 1 unidad de la camara
				t = tan( _deg2rad(self.fov) / 2 )
				r = t * self.width / self.height
				Px *= r
				Py *= t

				#Nuestra camara siempre esta viendo hacia -Z
				direction = V3(Px, Py, -1)
				direction = norm(direction)

				material = None

				for obj in self.scene:
					intersect = obj.ray_intersect(self.camPosition, direction)
					if intersect is not None:
						if intersect.distance < self.zbuffer[y][x]:
							self.zbuffer[y][x] = intersect.distance
							material = obj.material

				if material is not None:
					self.glVertex_coord(x, y, material.diffuse)

