#!/usr/bin/env python3
import os, sys, subprocess
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

def makeFibre(ele, azi,sectionRad = 0.02,fibreName = 'Fibre', use_bevel_object=False, use_mesh_torus=True, use_grease_pencil=True, bevel_factor=4, extrude_factor=10):
	#bezier curve controls cross section of the fibre for visualisation, zero in reality
	if use_bevel_object:
		ops.curve.primitive_bezier_circle_add()
		fibreCrossSec = bpy.context.active_object
		fibreCrossSec.name = fibreName +'_fibreCrossSec'
	
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
	else:
		ele /= 2
		baseRad = 2*((1/(pi/2 - ele)) - (2/pi))
		fibreRad = baseRad + 1/(baseRad +1)
		fibreCentre = (baseRad*sin(azi), baseRad*cos(azi),0)
		ops.curve.primitive_bezier_circle_add()
		fibre = bpy.context.active_object
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

		if use_mesh_torus:
			bpy.ops.mesh.primitive_torus_add(major_segments=12, minor_segments=12, major_radius=1, minor_radius=2)
			tor = bpy.context.active_object
			mod = tor.modifiers.new('sphere', type="CAST")
			mod.factor = -1.5
			tor.scale *= 0.25
			tor.parent = fibre
			tor.location.y = 1
			tor.data.materials.append(new_mat)
			if use_grease_pencil:
				bpy.ops.object.convert(target="GPENCIL")
				tor = bpy.context.active_object
				tor.data.materials[1].grease_pencil.show_fill=False
				tor.data.materials[1].grease_pencil.show_stroke=True
				#tor.data.materials[1].grease_pencil.color[3]=0.3
				bpy.ops.object.gpencil_modifier_add(type="GP_SUBDIV")
				mod = tor.grease_pencil_modifiers[-1]
				mod.level = 3
				#bpy.ops.object.gpencil_modifier_add(type="GP_DASH")
			else:
				mod = tor.modifiers.new('wire', type="WIREFRAME")
				tor.display_type = "WIRE"

	ops.object.select_all(action='DESELECT')
	return fibre

def mkhopf(tori = 6, fibresPerTorus = 50, section = 0.8):
	if 'Cube' in bpy.data.objects:
		bpy.data.objects.remove( bpy.data.objects['Cube'] )

	context.scene.cursor.location = (0.0, 0.0, 0.0)

	
	elevationRange = [(e+1)*pi/tori for e in range(tori)][:-1]
	azimuthRange = [(a*2*pi*section)/fibresPerTorus for a in range(fibresPerTorus)]
	
	for ele in elevationRange:
		for azi in azimuthRange:
			makeFibre(ele,azi,fibreName = 'Fibre_{}_{}'.format(ele,azi))

	makeFibre(0,0,fibreName = 'Fibre_0_0')
	makeFibre(pi,0,fibreName='Fibre_{}_0'.format(pi))


if __name__ == '__main__':
	args = []
	kwargs = {}
	for arg in sys.argv:
		if arg.startswith('--') and '=' in arg:
			args.append(arg)
			k,v = arg.split('=')
			k = k[2:]
			if k=='section':
				kwargs[k]=float(v)
			else:
				kwargs[k]=int(v)
	if not bpy:
		cmd = ['blender', '--python', __file__]
		if args:
			cmd += ['--'] + args
		print(cmd)
		subprocess.check_call(cmd)
		sys.exit()
	mkhopf(**kwargs)
	
	
