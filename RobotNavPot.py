# -*- coding: utf-8 -*-
"""
Way Point navigtion

(c) S. Bertrand
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Timer as tmr
import Potential


def normaliser_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def distance_points(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def creer_points_cercle(centre, rayon, pas_deg):
    angles = np.arange(0.0, 360.0, pas_deg)
    points = []
    for ang in angles:
        rad = math.radians(ang)
        x = centre[0] + rayon * math.cos(rad)
        y = centre[1] + rayon * math.sin(rad)
        points.append((x, y, ang))
    return points


def analyser_mesures_cercle(mesures, seuil):
    segments = []
    segment = []
    for mesure in mesures:
        if mesure[0] >= seuil:
            segment.append(mesure)
        else:
            if segment:
                segments.append(segment)
                segment = []
    if segment:
        segments.append(segment)

    if len(segments) > 1 and mesures:
        if mesures[0][0] >= seuil and mesures[-1][0] >= seuil:
            premier = segments[0]
            dernier = segments[-1]
            fusion = dernier + premier
            segments = [fusion] + segments[1:-1]

    resultats = []
    for seg in segments:
        angles_seg = []
        decalage = 0.0
        precedent = None
        for pot_mes, angle_deg, _ in seg:
            angle_norm = angle_deg % 360.0
            if precedent is not None and angle_norm < precedent:
                decalage += 360.0
            angles_seg.append(angle_norm + decalage)
            precedent = angle_norm
        if not angles_seg:
            continue
        angle_milieu = (angles_seg[0] + angles_seg[-1]) / 2.0
        angle_normalise = angle_milieu % 360.0

        meilleure_mesure = seg[0]
        meilleur_ecart = None
        for mesure in seg:
            angle_mes = mesure[1] % 360.0
            ecart = abs(((angle_mes - angle_normalise + 540.0) % 360.0) - 180.0)
            if meilleur_ecart is None or ecart < meilleur_ecart:
                meilleur_ecart = ecart
                meilleure_mesure = mesure

        resultats.append({
            'angle': angle_normalise,
            'mesure': meilleure_mesure,
            'segment': seg
        })
    return resultats


def estimer_centre_ligne(centre, pot_centre, mesure, pot_max=320.0):
    pot_mes, _, pos_mes = mesure
    dist = distance_points(centre, pos_mes)
    if dist == 0.0:
        return centre
    delta_centre = max(pot_max - pot_centre, 0.0)
    delta_mesure = max(pot_max - pot_mes, 0.0)
    denominateur = delta_centre + delta_mesure
    if denominateur == 0.0:
        return centre
    facteur = delta_centre / denominateur
    vecteur_x = pos_mes[0] - centre[0]
    vecteur_y = pos_mes[1] - centre[1]
    pos_estimee = (centre[0] + vecteur_x * facteur, centre[1] + vecteur_y * facteur)
    return pos_estimee


def fusionner_positions(liste_points, dist_max):
    points = list(liste_points)
    modifie = True
    while modifie and len(points) > 1:
        modifie = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if distance_points(points[i], points[j]) <= dist_max:
                    x = (points[i][0] + points[j][0]) / 2.0
                    y = (points[i][1] + points[j][1]) / 2.0
                    nouveau = (x, y)
                    points.pop(j)
                    points.pop(i)
                    points.append(nouveau)
                    modifie = True
                    break
            if modifie:
                break
    return points


def estimer_centre_zone(mesures):
    somme_poids = 0.0
    somme_x = 0.0
    somme_y = 0.0
    for pot_mesure, _, pos in mesures:
        somme_poids += pot_mesure
        somme_x += pot_mesure * pos[0]
        somme_y += pot_mesure * pos[1]
    if somme_poids == 0.0:
        return None
    return (somme_x / somme_poids, somme_y / somme_poids)

# saisie utilisateur
choix_difficulte = int(input("Choisir la difficulté (1, 2 ou 3) : "))
choix_hasard = input("Nuage aléatoire ? (T/F) : ").strip().upper() == 'T'


# robot
x0 = -20.0
y0 = -20.0
theta0 = np.pi/4.0
robot = rob.Robot(x0, y0, theta0)


# potential
pot = Potential.Potential(difficulty=choix_difficulte, random=choix_hasard)

# paramètres principaux
pot_seuil = 305.0
liste_rayon = [0.0, 10.0, 10.0, 18.0]
liste_fusion = [0.0, 10.0, 10.0, 16.0]
rayon_cercle = liste_rayon[choix_difficulte]
pas_angle = 10.0
dist_fusion = liste_fusion[choix_difficulte]
v_avance = 1.0
v_point = 0.8
tolerance_point = 0.5
tolerance_angle = 0.05
periode_mesure = 0.5
nb_zones = choix_difficulte

# stockage des mesures et états
mesures_trajet = []
file_points = []
zones_trouvees = []
mesures_cercle = []
points_cercle = []
indice_cercle = 0
centre_cercle = None
point_cible = None
etat = 'orientation'
direction_marche = math.atan2(-y0, -x0)
point_depart = (x0, y0)
point_courant = None
pot_point_courant = None
pot_centre_cercle = 0.0
retour_active = False


# position control loop: gain and timer
kpPos = 0.8
positionCtrlPeriod = 0.2#0.01
timerPositionCtrl = tmr.Timer(positionCtrlPeriod)

# orientation control loop: gain and timer
kpOrient = 2.5
orientationCtrlPeriod = 0.05#0.01
timerOrientationCtrl = tmr.Timer(orientationCtrlPeriod)



# list of way points list of [x coord, y coord]
WPlist = [ [x0,y0] ]
#threshold for change to next WP
epsilonWP = 0.2
# init WPManager
WPManager = rob.WPManager(WPlist, epsilonWP)


# duration of scenario and time step for numerical integration
t0 = 0.0
tf = 800.0
dt = 0.01
simu = rob.RobotSimulation(robot, t0, tf, dt)


# initialize control inputs
Vr = 0.0
thetar = robot.theta
omegar = 0.0

timerMesure = tmr.Timer(periode_mesure)

fin_simu = False
temps_simu = 0.0

# loop on simulation time
for t in simu.t:

    if fin_simu:
        break

    pot_actuel = pot.value([robot.x, robot.y])
    temps_simu = t

    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        if etat == 'orientation':
            thetar = math.atan2(-robot.y, -robot.x)
            Vr = 0.0
            if abs(normaliser_angle(robot.theta - thetar)) < tolerance_angle:
                etat = 'avance'
                direction_marche = thetar

        elif etat == 'avance':
            if file_points and not retour_active:
                point_cible = file_points.pop(0)
                etat = 'aller_point_detection'
                Vr = 0.0
            else:
                thetar = direction_marche
                Vr = v_avance
                WPManager.xr = robot.x + math.cos(thetar)
                WPManager.yr = robot.y + math.sin(thetar)
                if timerMesure.isEllapsed(t):
                    mesures_trajet.append((t, robot.x, robot.y, pot_actuel))
                if pot_actuel >= pot_seuil:
                    point_courant = (robot.x, robot.y)
                    pot_point_courant = pot_actuel
                    etat = 'initialiser_cercle'
                    Vr = 0.0
                    timerMesure.reset(t)
                else:
                    if abs(robot.x) >= 24.0 or abs(robot.y) >= 24.0:
                        direction_marche = normaliser_angle(direction_marche + math.pi / 2.0)

        elif etat == 'aller_point_detection':
            thetar = math.atan2(point_cible[1] - robot.y, point_cible[0] - robot.x)
            dist_point = distance_points((robot.x, robot.y), point_cible)
            if dist_point <= tolerance_point:
                point_courant = point_cible
                pot_point_courant = pot_actuel
                etat = 'initialiser_cercle'
                Vr = 0.0
                timerMesure.reset(t)
            else:
                Vr = v_point
            WPManager.xr = point_cible[0]
            WPManager.yr = point_cible[1]

        elif etat == 'initialiser_cercle':
            centre_cercle = point_courant
            if pot_point_courant is None:
                pot_centre_cercle = pot.value([centre_cercle[0], centre_cercle[1]])
            else:
                pot_centre_cercle = pot_point_courant
            points_cercle = creer_points_cercle(centre_cercle, rayon_cercle, pas_angle)
            indice_cercle = 0
            mesures_cercle = []
            if points_cercle:
                point_cible = (points_cercle[0][0], points_cercle[0][1])
                thetar = math.atan2(point_cible[1] - robot.y, point_cible[0] - robot.x)
                Vr = v_point
                etat = 'aller_point_cercle'
                WPManager.xr = point_cible[0]
                WPManager.yr = point_cible[1]
            else:
                etat = 'analyse_cercle'
                Vr = 0.0

        elif etat == 'aller_point_cercle':
            point_temp = points_cercle[indice_cercle]
            point_cible = (point_temp[0], point_temp[1])
            thetar = math.atan2(point_cible[1] - robot.y, point_cible[0] - robot.x)
            dist_point = distance_points((robot.x, robot.y), point_cible)
            Vr = v_point
            WPManager.xr = point_cible[0]
            WPManager.yr = point_cible[1]
            if dist_point <= tolerance_point:
                pot_point = pot_actuel
                angle_point = point_temp[2]
                mesures_cercle.append((pot_point, angle_point, (robot.x, robot.y)))
                indice_cercle += 1
                if indice_cercle >= len(points_cercle):
                    etat = 'analyse_cercle'
                    Vr = 0.0
                else:
                    prochain = points_cercle[indice_cercle]
                    point_cible = (prochain[0], prochain[1])
                    thetar = math.atan2(point_cible[1] - robot.y, point_cible[0] - robot.x)

        elif etat == 'analyse_cercle':
            Vr = 0.0
            detections = analyser_mesures_cercle(mesures_cercle, pot_seuil)
            points_a_explorer = []
            zones_a_ajouter = []
            for detection in detections:
                mesure_ref = detection['mesure']
                pos_mes = mesure_ref[2]
                dist_mesure = distance_points(centre_cercle, pos_mes)
                if dist_mesure <= dist_fusion:
                    pos_estimee = estimer_centre_ligne(centre_cercle, pot_centre_cercle, mesure_ref)
                    zones_a_ajouter.append(pos_estimee)
                else:
                    points_a_explorer.append(pos_mes)

            if points_a_explorer:
                points_a_explorer = fusionner_positions(points_a_explorer, dist_fusion)
                for pt in points_a_explorer:
                    proche = False
                    for zone in zones_trouvees:
                        if distance_points(zone, pt) <= dist_fusion:
                            proche = True
                            break
                    if proche:
                        continue
                    deja = False
                    for cible in file_points:
                        if distance_points(cible, pt) <= tolerance_point:
                            deja = True
                            break
                    if not deja:
                        file_points.append(pt)

            if zones_a_ajouter:
                for zone_cand in zones_a_ajouter:
                    valide = True
                    for zone in zones_trouvees:
                        if distance_points(zone, zone_cand) <= dist_fusion:
                            valide = False
                            break
                    if valide:
                        zones_trouvees.append(zone_cand)

            if not detections:
                centre_estime = estimer_centre_zone(mesures_cercle)
                if centre_estime is None:
                    centre_estime = centre_cercle
                valide = True
                for zone in zones_trouvees:
                    if distance_points(zone, centre_estime) <= dist_fusion:
                        valide = False
                        break
                if valide:
                    zones_trouvees.append(centre_estime)

            if len(zones_trouvees) >= nb_zones and not retour_active:
                retour_active = True
                etat = 'retour'
            elif file_points and not retour_active:
                point_cible = file_points.pop(0)
                etat = 'aller_point_detection'
            else:
                etat = 'avance'

            mesures_cercle = []
            points_cercle = []
            pot_point_courant = None
            pot_centre_cercle = 0.0

        elif etat == 'retour':
            point_cible = point_depart
            thetar = math.atan2(point_cible[1] - robot.y, point_cible[0] - robot.x)
            dist_retour = distance_points((robot.x, robot.y), point_depart)
            if dist_retour <= 2.0:
                Vr = 0.0
                fin_simu = True
            else:
                Vr = v_point
            WPManager.xr = point_cible[0]
            WPManager.yr = point_cible[1]

        if math.fabs(robot.theta - thetar) > math.pi:
            thetar = thetar + math.copysign(2 * math.pi, robot.theta)

    # orientation control loop
    if timerOrientationCtrl.isEllapsed(t):
    
        erreur_theta = normaliser_angle(robot.theta - thetar)
        omegar = -kpOrient * erreur_theta

    # assign control inputs to robot
    robot.setV(Vr)
    robot.setOmega(omegar)

    # integrate motion
    robot.integrateMotion(dt)

    # store data to be plotted
    simu.addData(robot, WPManager, Vr, thetar, omegar, pot.value([robot.x,robot.y]))

# end of loop on simulation time

print("Zones trouvées :", zones_trouvees)
print("Temps de simulation :", round(temps_simu, 2), "s")


# close all figures
plt.close("all")

# generate plots
fig,ax = simu.plotXY(1)
pot.plot(noFigure=None, fig=fig, ax=ax)  # plot potential for verification of solution
if zones_trouvees:
    x_zones = [zone[0] for zone in zones_trouvees]
    y_zones = [zone[1] for zone in zones_trouvees]
    ax.scatter(x_zones, y_zones, c='red', marker='o', s=60, label='Centre trouvé')
    ax.legend()

simu.plotXYTheta(2)
#simu.plotVOmega(3)

simu.plotPotential(4)



simu.plotPotential3D(5)


# show plots
plt.show()





# # Animation *********************************
# fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-25, 25), ylim=(-25, 25))
# ax.grid()
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')

# robotBody, = ax.plot([], [], 'o-', lw=2)
# robotDirection, = ax.plot([], [], '-', lw=1, color='k')
# wayPoint, = ax.plot([], [], 'o-', lw=2, color='b')
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# potential_template = 'potential = %.1f'
# potential_text = ax.text(0.05, 0.1, '', transform=ax.transAxes)
# WPArea, = ax.plot([], [], ':', lw=1, color='b')

# thetaWPArea = np.arange(0.0,2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
# xWPArea = WPManager.epsilonWP*np.cos(thetaWPArea)
# yWPArea = WPManager.epsilonWP*np.sin(thetaWPArea)

# def initAnimation():
#     robotDirection.set_data([], [])
#     robotBody.set_data([], [])
#     wayPoint.set_data([], [])
#     WPArea.set_data([], [])
#     robotBody.set_color('r')
#     robotBody.set_markersize(20)    
#     time_text.set_text('')
#     potential_text.set_text('')
#     return robotBody,robotDirection, wayPoint, time_text, potential_text, WPArea  
    
# def animate(i):  
#     robotBody.set_data(simu.x[i], simu.y[i])          
#     wayPoint.set_data(simu.xr[i], simu.yr[i])
#     WPArea.set_data(simu.xr[i]+xWPArea.transpose(), simu.yr[i]+yWPArea.transpose())    
#     thisx = [simu.x[i], simu.x[i] + 0.5*math.cos(simu.theta[i])]
#     thisy = [simu.y[i], simu.y[i] + 0.5*math.sin(simu.theta[i])]
#     robotDirection.set_data(thisx, thisy)
#     time_text.set_text(time_template%(i*simu.dt))
#     potential_text.set_text(potential_template%(pot.value([simu.x[i],simu.y[i]])))
#     return robotBody,robotDirection, wayPoint, time_text, potential_text, WPArea

# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(simu.t)),
#     interval=4, blit=True, init_func=initAnimation, repeat=False)
# #interval=25

# #ani.save('robot.mp4', fps=15)

