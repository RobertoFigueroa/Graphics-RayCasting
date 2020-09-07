from render import Raytracer, V3, V2, color
from utils import color
from sphere import Sphere, Material

brick = Material(diffuse = color(0.8, 0.25, 0.25 ))
stone = Material(diffuse = color(0.4, 0.4, 0.4 ))
grass = Material(diffuse = color(0.5, 1, 0))
snow = Material(diffuse= color(1,1,1))
dot = Material(diffuse= color(0,0,0))
carrot = Material(diffuse= color(1,0.6,0))

width = 500
height = 300
r = Raytracer(width, height)

r.scene.append( Sphere(V3(0, 1.7, -10), 1, snow) )
r.scene.append( Sphere(V3(0, 0.4,  -10), 1.2, snow) )
r.scene.append( Sphere(V3(0, -1.1, -10), 1.6, snow) )
r.scene.append( Sphere(V3(0, 0.9, -5), 0.1, carrot) )
r.scene.append( Sphere(V3(0, 0.3, -5), 0.1, stone) )
r.scene.append( Sphere(V3(0, -0.2, -5), 0.1, stone) )
r.scene.append( Sphere(V3(0, -0.7, -5), 0.1, stone) )

r.scene.append( Sphere(V3(-0.1, 1, -5), 0.1, stone) )
r.scene.append( Sphere(V3(0.1, 1, -5), 0.1, stone) )

r.scene.append( Sphere(V3(-0.2, 0.7, -5), 0.08, stone) )
r.scene.append( Sphere(V3(0.2, 0.7, -5), 0.08, stone) )
r.scene.append( Sphere(V3(0, 0.6, -5), 0.08, stone) )



 
r.rtRender()


r.glFinish('output.bmp')
