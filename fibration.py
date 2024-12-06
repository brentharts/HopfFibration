#!/usr/bin/env python3
# example:
# python3 fibration.py --fibresPerTorus=3 --tori=3 --spacetime=1 --gizmo=1


import os, sys, subprocess, math
from math import pi, atan2,cos,sin
try:
	import bpy
except:
	bpy = None
if bpy:
	from bpy import ops,data,context

def deleteAll():
	ops.object.select_all(action='DESELECT')
	for object in data.objects:
		object.select_set(True)
		ops.object.delete()
	for material in data.materials:
		material.user_clear()
		data.materials.remove(material)

def HSV2RGB(H,S,V=1):
	C = V*S
	X = C*(1-abs((H/60)%2 - 1))
	m = V-C
	if H <60:
		RGB_dash = [C,X,0]
	elif H<120:
		RGB_dash = [X,C,0]
	elif H<180:
		RGB_dash = [0,C,X]
	elif H<240:
		RGB_dash = [0,X,C]
	elif H<300:
		RGB_dash = [X,0,C]
	else:
		RGB_dash = [C,0,X]
	Rd,Gd,Bd = RGB_dash
	return [(Rd+m),(Gd+m),(Bd+m)]

def makeFibre(ele, azi,sectionRad = 0.02,fibreName = 'Fibre', use_bevel_object=False, use_flare=False, use_mesh_torus=True, use_mesh_twist=False, use_grease_pencil=True, bevel_factor=4, extrude_factor=10, parent=None):
	#bezier curve controls cross section of the fibre for visualisation, zero in reality
	if use_bevel_object:
		ops.curve.primitive_bezier_circle_add()
		fibreCrossSec = bpy.context.active_object
		fibreCrossSec.name = fibreName +'_fibreCrossSec'
		fibreCrossSec.parent = parent
	
	if ele == pi:
		ops.curve.primitive_nurbs_path_add()
		fibre = bpy.context.active_object
		fibre.rotation_euler.y = pi/2
		fibre.scale.x = 1000
		if use_bevel_object:
			fibreCrossSec.scale = [sectionRad,sectionRad,1]
			fibre.data.bevel_object = fibreCrossSec
		else:
			fibre.data.bevel_depth = sectionRad * bevel_factor
		new_mat = data.materials.new(name = 'FibreColour')
		new_mat.diffuse_color = (0,0,1,1)
		fibre.data.materials.append(new_mat)
		fibre.parent = parent
	else:
		ele /= 2
		baseRad = 2*((1/(pi/2 - ele)) - (2/pi))
		fibreRad = baseRad + 1/(baseRad +1)
		fibreCentre = (baseRad*sin(azi), baseRad*cos(azi),0)
		ops.curve.primitive_bezier_circle_add()
		fibre = bpy.context.active_object
		fibre.parent = parent
		fibre.name, fibre.data.name = (fibreName,fibreName)
		fibre.location = fibreCentre
		fibre.scale = (fibreRad,fibreRad,fibreRad)
		fibre.rotation_euler.y = ele
		fibre.rotation_euler.z = -azi
		if use_bevel_object:
			fibreCrossSec.scale = [sectionRad/fibreRad,sectionRad/fibreRad,1]
			fibre.data.bevel_mode = "OBJECT"
			fibre.data.bevel_object = fibreCrossSec
		else:
			fibre.data.bevel_depth = (sectionRad/fibreRad) * bevel_factor
			fibre.data.extrude = (sectionRad/fibreRad) * extrude_factor
		new_mat = data.materials.new(name = 'FibreColour')
		hue = (azi*15/pi) + ele*330/pi
		r,g,b = HSV2RGB(hue,1,1)
		new_mat.diffuse_color = (r,g,b,1)
		fibre.data.materials.append(new_mat)

		if use_flare:
			fibre.data.splines[0].bezier_points[1].radius = 10
			fibre.data.splines[0].bezier_points[1].tilt = math.radians(100)
			fibre.data.splines[0].bezier_points[0].tilt = math.radians(-100)

		if use_mesh_torus:
			bpy.ops.mesh.primitive_torus_add(major_segments=12, minor_segments=12, major_radius=1, minor_radius=2)
			tor = bpy.context.active_object
			mod = tor.modifiers.new('sphere', type="CAST")
			mod.factor = -1.5
			tor.scale *= 0.25
			#tor.scale *= 0.125
			tor.parent = fibre
			tor.location.y = 1
			tor.data.materials.append(new_mat)

			if use_mesh_twist:
				mod = tor.modifiers.new('twist', type="SIMPLE_DEFORM")
				mod.factor = (hue / 45)
				mod.deform_axis = "Z"
				mod = tor.modifiers.new('subsurf', type="SUBSURF")

			if use_grease_pencil:
				bpy.ops.object.convert(target="GPENCIL")
				tor = bpy.context.active_object
				mat = tor.data.materials[1]
				#mat.grease_pencil.show_fill=False
				mat.grease_pencil.use_fill_holdout=True
				mat.grease_pencil.fill_color[3]=0.0
				mat.grease_pencil.show_stroke=True
				#tor.data.materials[1].grease_pencil.color[3]=0.3
				bpy.ops.object.gpencil_modifier_add(type="GP_SUBDIV")
				mod = tor.grease_pencil_modifiers[-1]
				mod.level = 3
				#bpy.ops.object.gpencil_modifier_add(type="GP_DASH")
				tor.parent = parent
			else:
				mod = tor.modifiers.new('wire', type="WIREFRAME")
				tor.display_type = "WIRE"

	ops.object.select_all(action='DESELECT')
	return fibre


def create_linear_curve(points):
	curve_data = bpy.data.curves.new(name="LinearCurve", type='CURVE')
	curve_data.dimensions = '3D'
	polyline = curve_data.splines.new('POLY')
	polyline.points.add(len(points) - 1)
	for i, point in enumerate(points):
		x,y,z = point
		polyline.points[i].co.x = x
		polyline.points[i].co.y = y
		polyline.points[i].co.z = z
		#polyline.points[i].tilt = i*30

	#polyline.points[0].tilt = math.radians(90)
	#polyline.points[1].tilt = math.radians(180)
	polyline.points[1].radius = 0
	polyline.points[0].radius = 10
	polyline.points[-1].radius = 10

	curve_obj = bpy.data.objects.new("LinearCurveObject", curve_data)
	bpy.context.collection.objects.link(curve_obj)
	curve_obj.data.extrude = 0.4
	return curve_obj


def mkhopf(tori = 6, fibresPerTorus = 50, section = 0.8, spacetime=False, flare=False, gizmo=False):
	if 'Cube' in bpy.data.objects:
		bpy.data.objects.remove( bpy.data.objects['Cube'] )

	context.scene.cursor.location = (0.0, 0.0, 0.0)

	bpy.ops.object.empty_add()
	root = bpy.context.active_object
	root.name='HOPF'
	elevationRange = [(e+1)*pi/tori for e in range(tori)][:-1]
	azimuthRange = [(a*2*pi*section)/fibresPerTorus for a in range(fibresPerTorus)]

	fibers = [(0,0,50), (0,0,0)]
	for ele in elevationRange:
		for azi in azimuthRange:
			f = makeFibre(
					ele,azi,
					fibreName = 'Fibre_{}_{}'.format(ele,azi), 
					parent=root,
					use_flare=flare,
					use_mesh_torus=spacetime,
				)
			fibers.append(f.location)

	f=makeFibre(0,0,fibreName = 'Fibre_0_0', parent=root)
	f=makeFibre(pi,0,fibreName='Fibre_{}_0'.format(pi), parent=root)

	if gizmo:
		x,y,z = fibers[-1]
		fibers.append((x,y+50,z))
		cu = create_linear_curve(fibers)
		cu.show_in_front=True
		mod = cu.modifiers.new('solid', type="SOLIDIFY")
		mod.thickness = 0.1
		mod.use_rim = False
		mod.material_offset = 1
		mat = bpy.data.materials.new(name='A')
		mat.diffuse_color= [1,0,0,1]
		cu.data.materials.append(mat)
		mat = bpy.data.materials.new(name='B')
		mat.diffuse_color= [0,0,1,1]
		cu.data.materials.append(mat)
		cu.parent = root

	return root

if __name__ == '__main__':
	args = []
	kwargs = {}
	blend = None
	for arg in sys.argv:
		if arg.startswith('--') and '=' in arg:
			args.append(arg)
			k,v = arg.split('=')
			k = k[2:]
			if k=='section':
				kwargs[k]=float(v)
			else:
				kwargs[k]=int(v)
		elif arg.endswith('.blend'):
			blend = arg
	if not bpy:
		cmd = ['blender']
		if blend: cmd.append(blend)
		cmd += ['--python', __file__]
		if args:
			cmd += ['--'] + args
		print(cmd)
		subprocess.check_call(cmd)
		sys.exit()
	mkhopf(**kwargs)
	
	
