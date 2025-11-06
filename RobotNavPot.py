# -*- coding: utf-8 -*-

import math
from typing import Optional
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import Timer as tmr
import Potential


MIN_COORD_MATRICE = -25
MAX_COORD_MATRICE = 25
TAILLE_MATRICE = MAX_COORD_MATRICE - MIN_COORD_MATRICE + 1
RAYON_ACTIVATION_NAVIGATION = 10.0
MAX_CERCLES_CONSECUTIFS = 2

RAYON_RECONSTRUCTION_BASE = 8.0
LARGEUR_COURONNE_RECONSTRUCTION = 4.0

class ParametresCercles:
    seuil_ratio: float = 0.98
    pourc_dist: float = 70.0
    facteur_min_reference: float = 0.6
    rayon_min: float = 2.5
    largeur_ratio: float = 1
    largeur_min: float = 2.0

def configurer_reconstruction_cercles(*, rayon_base: Optional[float] = None, largeur_couronne: Optional[float] = None):

    global RAYON_RECONSTRUCTION_BASE, LARGEUR_COURONNE_RECONSTRUCTION

    if rayon_base is not None:
        RAYON_RECONSTRUCTION_BASE = float(rayon_base)
    if largeur_couronne is not None:
        LARGEUR_COURONNE_RECONSTRUCTION = float(largeur_couronne)

    return RAYON_RECONSTRUCTION_BASE, LARGEUR_COURONNE_RECONSTRUCTION


def configurer_parametres_cercles(**kwargs):

    global PARAMETRES_CERCLES

    for cle, valeur in kwargs.items():
        setattr(PARAMETRES_CERCLES, cle, float(valeur))

    return PARAMETRES_CERCLES


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


def point_dans_cercle(point, centre, rayon, marge=1e-6):
    if point is None:
        return False
    return distance_points(point, centre) <= rayon + marge


def point_dans_anciens_cercles(point, cercles):
    if point is None:
        return False
    for cercle in cercles:
        if point_dans_cercle(point, cercle['centre'], cercle['rayon']):
            return True
    return False


def nettoyer_file_points(file_points, cercles):
    if not file_points:
        return
    file_points[:] = [pt for pt in file_points if not point_dans_anciens_cercles(pt, cercles)]


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


def arrondir_coord(valeur):
    if valeur >= 0.0:
        return int(math.floor(valeur + 0.5))
    return int(math.ceil(valeur - 0.5))


def mettre_a_jour_matrice(matrice, x, y, potentiel):
    ix = arrondir_coord(x)
    iy = arrondir_coord(y)
    if (
        ix < MIN_COORD_MATRICE
        or ix > MAX_COORD_MATRICE
        or iy < MIN_COORD_MATRICE
        or iy > MAX_COORD_MATRICE
    ):
        return False
    idx_x = ix - MIN_COORD_MATRICE
    idx_y = iy - MIN_COORD_MATRICE
    valeur_actuelle = matrice[idx_x][idx_y]
    if valeur_actuelle == 0.0:
        matrice[idx_x][idx_y] = potentiel
    else:
        matrice[idx_x][idx_y] = (valeur_actuelle + potentiel) / 2.0
    return True


def trouver_plus_gros_carre_inexplore(matrice):
    if not matrice:
        return None
    taille_x = len(matrice)
    taille_y = len(matrice[0]) if matrice[0] else 0
    if taille_y == 0:
        return None

    dp = [[0 for _ in range(taille_y)] for _ in range(taille_x)]
    meilleure_taille = 0
    meilleur_depart = None

    for i in reversed(range(taille_x)):
        for j in reversed(range(taille_y)):
            if matrice[i][j] != 0.0:
                dp[i][j] = 0
                continue
            taille_case = 1
            if i + 1 < taille_x and j + 1 < taille_y:
                taille_case += min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
            dp[i][j] = taille_case
            if taille_case > meilleure_taille:
                meilleure_taille = taille_case
                meilleur_depart = (i, j)

    if meilleure_taille == 0 or meilleur_depart is None:
        return None

    depart_i, depart_j = meilleur_depart
    centre_i = depart_i + (meilleure_taille - 1) / 2.0
    centre_j = depart_j + (meilleure_taille - 1) / 2.0
    coord_x = MIN_COORD_MATRICE + centre_i
    coord_y = MIN_COORD_MATRICE + centre_j
    return (arrondir_coord(coord_x), arrondir_coord(coord_y))


def calculer_direction_depuis_mesures(mesures):
    if len(mesures) < 3:
        return None
    echantillons = mesures[-3:]
    matrice_systeme = np.array([[x, y, 1.0] for x, y, _ in echantillons], dtype=float)
    vecteur_potentiels = np.array([potentiel for _, _, potentiel in echantillons], dtype=float)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(matrice_systeme, vecteur_potentiels, rcond=None)
    except np.linalg.LinAlgError:
        return None
    grad_x, grad_y = coeffs[0], coeffs[1]
    norme = math.hypot(grad_x, grad_y)
    if norme == 0.0:
        return None
    return math.atan2(grad_y, grad_x)


def extraire_mesures_matrice(matrice):
    mesures = []
    for ix in range(len(matrice)):
        for iy in range(len(matrice[0])):
            valeur = matrice[ix][iy]
            if valeur <= 0.0:
                continue
            x = MIN_COORD_MATRICE + ix
            y = MIN_COORD_MATRICE + iy
            mesures.append((x, y, valeur))
    return mesures


def valeur_matrice_coord(matrice, coord):
    ix = arrondir_coord(coord[0])
    iy = arrondir_coord(coord[1])
    if (
        ix < MIN_COORD_MATRICE
        or ix > MAX_COORD_MATRICE
        or iy < MIN_COORD_MATRICE
        or iy > MAX_COORD_MATRICE
    ):
        return 0.0
    idx_x = ix - MIN_COORD_MATRICE
    idx_y = iy - MIN_COORD_MATRICE
    return matrice[idx_x][idx_y]

def determiner_valeur_centre(matrice, centre, mesures, seuil_detection):
    valeur = valeur_matrice_coord(matrice, centre)
    if valeur > 0.0:
        return valeur
    if not mesures:
        return seuil_detection
    rayon_recherche = 2.5
    valeurs_proches = [
        pot for (mx, my, pot) in mesures if distance_points((mx, my), centre) <= rayon_recherche
    ]
    if valeurs_proches:
        return max(valeurs_proches)
    return seuil_detection


def calculer_intensite_cercles(distances, valeur_centre, baseline):
    rayon_fort = RAYON_RECONSTRUCTION_BASE
    largeur_couronne = LARGEUR_COURONNE_RECONSTRUCTION
    rayon_intermediaire = rayon_fort + largeur_couronne
    rayon_large = rayon_intermediaire + largeur_couronne

    valeur_intermediaire = max(valeur_centre * 0.65, baseline + 1.0)
    valeur_exterieure = max(valeur_centre * 0.35, baseline)

    intensite = np.full_like(distances, baseline, dtype=float)

    masque_fort = distances <= rayon_fort
    intensite[masque_fort] = valeur_centre

    masque_inter = (distances > rayon_fort) & (distances <= rayon_intermediaire)
    if np.any(masque_inter):
        proportion = (distances[masque_inter] - rayon_fort) / max(largeur_couronne, 1e-9)
        intensite[masque_inter] = valeur_centre - (
            valeur_centre - valeur_intermediaire
        ) * proportion

    masque_large = (distances > rayon_intermediaire) & (distances <= rayon_large)
    if np.any(masque_large):
        proportion = (distances[masque_large] - rayon_intermediaire) / max(largeur_couronne, 1e-9)
        intensite[masque_large] = valeur_intermediaire - (
            valeur_intermediaire - valeur_exterieure
        ) * proportion

    return (
        intensite,
        rayon_fort,
        rayon_intermediaire,
        rayon_large,
        valeur_centre,
        valeur_intermediaire,
        valeur_exterieure,
    )


def reconstruire_carte_emissions(
    matrice,
    seuil_detection,
    zones_reference=None,
    rayon_detection=None,
    mesures=None,
):

    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.linspace(MIN_COORD_MATRICE, MAX_COORD_MATRICE, TAILLE_MATRICE)
    y = np.linspace(MIN_COORD_MATRICE, MAX_COORD_MATRICE, TAILLE_MATRICE)
    X, Y = np.meshgrid(x, y, indexing='ij')

    mesures_matrice = extraire_mesures_matrice(matrice)
    valeurs_non_nulles = [mes[2] for mes in mesures_matrice]
    if valeurs_non_nulles:
        baseline = min(valeurs_non_nulles)
    else:
        baseline = seuil_detection * 0.5

    champ = np.full_like(X, baseline, dtype=float)
    distance_assignation = np.full_like(X, np.inf, dtype=float)

    cercles_trace = []

    if zones_reference:
        for centre in zones_reference:
            valeur_centre = determiner_valeur_centre(
                matrice, centre, mesures, seuil_detection
            )
            distances = np.hypot(X - centre[0], Y - centre[1])
            (
                intensite,
                rayon_fort,
                rayon_intermediaire,
                rayon_large,
                valeur_forte,
                valeur_moyenne,
                valeur_faible,
            ) = calculer_intensite_cercles(distances, valeur_centre, baseline)

            masque_proche = distances < distance_assignation
            champ[masque_proche] = intensite[masque_proche]
            distance_assignation[masque_proche] = distances[masque_proche]

            cercles_trace.append(
                (
                    centre,
                    rayon_fort,
                    rayon_intermediaire,
                    rayon_large,
                    valeur_forte,
                    valeur_moyenne,
                    valeur_faible,
                )
            )

    vmax = champ.max()
    niveaux = np.linspace(baseline, vmax, 20)
    cmap = plt.get_cmap('YlOrRd')

    contour = ax.contourf(
        X,
        Y,
        champ,
        levels=niveaux,
        cmap=cmap,
        extend='both',
    )

    if zones_reference:
        ax.scatter(
            [pt[0] for pt in zones_reference],
            [pt[1] for pt in zones_reference],
            c='black',
            marker='x',
            s=55,
            label='Centre détecté',
        )
        for centre, r_fort, r_inter, r_large, _, _, _ in cercles_trace:
            cercle_fort = Circle(
                centre,
                r_fort,
                fill=False,
                color='#d73027',
                linewidth=1.5,
                label='Zone forte',
            )
            cercle_moyen = Circle(
                centre,
                r_inter,
                fill=False,
                color='#fc8d59',
                linewidth=1.1,
                linestyle='--',
                label='Zone intermédiaire',
            )
            cercle_large = Circle(
                centre,
                r_large,
                fill=False,
                color='#fee08b',
                linewidth=1.0,
                linestyle=':',
                label='Zone faible',
            )
            ax.add_patch(cercle_large)
            ax.add_patch(cercle_moyen)
            ax.add_patch(cercle_fort)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right')
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Potentiel estimé')


    ax.set_title("Carte reconstruite des zones d'émission")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(MIN_COORD_MATRICE, MAX_COORD_MATRICE)
    ax.set_ylim(MIN_COORD_MATRICE, MAX_COORD_MATRICE)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.text(
        0.02,
        0.02,
        f'Seuil mission : {seuil_detection:.1f}',
        transform=ax.transAxes,
        color='white',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
    )
    return fig, ax


# saisie utilisateur
choix_difficulte = int(input("Choisir la difficulté (1, 2 ou 3) : "))
choix_hasard = input("Nuage aléatoire ? (T/F) : ").strip().upper() == 'T'

# potential
pot = Potential.Potential(difficulty=choix_difficulte, random=choix_hasard)


# robot
x0 = -20.0
y0 = -20.0
theta0 = np.pi/4.0
robot = rob.Robot(x0, y0, theta0)


# paramètres principaux
pot_seuil = 300.0
liste_rayon = [0.0, 11.0, 9.0, 10.0]
liste_fusion = [0.0, 11.0, 9.0, 10.0]
rayon_cercle = liste_rayon[choix_difficulte]
pas_angle = 10.0
dist_fusion = liste_fusion[choix_difficulte]
v_avance = 1.0
v_point = 0.8
tolerance_point = 0.5
periode_mesure = 0.25
nb_zones = choix_difficulte

# stockage des mesures et états
matrice_mesures = [[0.0 for _ in range(TAILLE_MATRICE)] for _ in range(TAILLE_MATRICE)]
mesures_enregistrees = []
file_points = []
zones_trouvees = []
mesures_cercle = []
points_cercle = []
indice_cercle = 0
centre_cercle = None
point_cible = None
point_objectif = None
etat = 'aller_point_objectif'
direction_marche = robot.theta
point_depart = (x0, y0)
point_courant = None
pot_point_courant = None
pot_centre_cercle = 0.0
retour_active = False
cercles_explores = []
cible_depuis_fusion = False
historique_mesures_avance = []
navigation_intelligente_active = False
compteur_gradient_nul = 0
compteur_cercles_consecutifs = 0

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
tf = 1200.0
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

        if etat == 'avance':
            if not navigation_intelligente_active and not retour_active:
                if not file_points:
                    etat = 'aller_point_objectif'
                    point_objectif = None
                    navigation_intelligente_active = False
                    compteur_gradient_nul = 0
                    historique_mesures_avance.clear()
                    compteur_cercles_consecutifs = 0
                    continue
            if file_points and not retour_active:
                nettoyer_file_points(file_points, cercles_explores)
            if file_points and not retour_active:
                point_cible = file_points.pop(0)
                cible_depuis_fusion = False
                etat = 'aller_point_detection'
                Vr = 0.0
                historique_mesures_avance.clear()
                compteur_gradient_nul = 0
            else:
                thetar = direction_marche
                Vr = v_avance
                WPManager.xr = robot.x + math.cos(thetar)
                WPManager.yr = robot.y + math.sin(thetar)
                if timerMesure.isEllapsed(t):
                    if mettre_a_jour_matrice(matrice_mesures, robot.x, robot.y, pot_actuel):
                        mesures_enregistrees.append((robot.x, robot.y, pot_actuel))
                    if navigation_intelligente_active:
                        historique_mesures_avance.append((robot.x, robot.y, pot_actuel))
                        if len(historique_mesures_avance) > 3:
                            historique_mesures_avance.pop(0)
                        nouvelle_direction = calculer_direction_depuis_mesures(historique_mesures_avance)
                        if nouvelle_direction is not None:
                            direction_marche = normaliser_angle(nouvelle_direction)
                            thetar = direction_marche
                            WPManager.xr = robot.x + math.cos(thetar)
                            WPManager.yr = robot.y + math.sin(thetar)
                            compteur_gradient_nul = 0
                        else:
                            compteur_gradient_nul += 1
                            if (
                                compteur_gradient_nul >= 5
                                and not retour_active
                                and not file_points
                            ):
                                navigation_intelligente_active = False
                                etat = 'aller_point_objectif'
                                point_objectif = None
                                historique_mesures_avance.clear()
                                timerMesure.reset(t)
                                continue
                    elif historique_mesures_avance:
                        historique_mesures_avance.clear()
                if pot_actuel >= pot_seuil:
                    point_courant = (robot.x, robot.y)
                    pot_point_courant = pot_actuel
                    etat = 'initialiser_cercle'
                    Vr = 0.0
                    historique_mesures_avance.clear()
                    timerMesure.reset(t)
                else:
                    if abs(robot.x) >= 24.0 or abs(robot.y) >= 24.0:
                        direction_marche = normaliser_angle(direction_marche + math.pi / 2.0)
                        thetar = direction_marche
                        WPManager.xr = robot.x + math.cos(thetar)
                        WPManager.yr = robot.y + math.sin(thetar)

        elif etat == 'aller_point_objectif':
            navigation_intelligente_active = False
            compteur_gradient_nul = 0
            compteur_cercles_consecutifs = 0
            if historique_mesures_avance:
                historique_mesures_avance.clear()
            if file_points and not retour_active:
                nettoyer_file_points(file_points, cercles_explores)
            if file_points and not retour_active:
                point_cible = file_points.pop(0)
                cible_depuis_fusion = False
                etat = 'aller_point_detection'
                Vr = 0.0
                historique_mesures_avance.clear()
                continue

            if timerMesure.isEllapsed(t):
                if mettre_a_jour_matrice(matrice_mesures, robot.x, robot.y, pot_actuel):
                    mesures_enregistrees.append((robot.x, robot.y, pot_actuel))

            if pot_actuel >= pot_seuil:
                point_courant = (robot.x, robot.y)
                pot_point_courant = pot_actuel
                etat = 'initialiser_cercle'
                Vr = 0.0
                historique_mesures_avance.clear()
                point_objectif = None
                compteur_gradient_nul = 0
                timerMesure.reset(t)
                continue

            if point_objectif is None:
                resultat_carre = trouver_plus_gros_carre_inexplore(matrice_mesures)
                if resultat_carre is not None:
                    point_objectif = (float(resultat_carre[0]), float(resultat_carre[1]))

            if point_objectif is not None:
                thetar = math.atan2(point_objectif[1] - robot.y, point_objectif[0] - robot.x)
                direction_marche = thetar
                WPManager.xr = point_objectif[0]
                WPManager.yr = point_objectif[1]
                dist_obj = distance_points((robot.x, robot.y), point_objectif)
                Vr = v_avance
                if dist_obj <= tolerance_point:
                    if mettre_a_jour_matrice(matrice_mesures, robot.x, robot.y, pot_actuel):
                        mesures_enregistrees.append((robot.x, robot.y, pot_actuel))
                    point_objectif = None
                if dist_obj <= RAYON_ACTIVATION_NAVIGATION:
                    navigation_intelligente_active = True
                    compteur_gradient_nul = 0
                    etat = 'avance'
                    direction_marche = robot.theta
                    point_objectif = None
                    historique_mesures_avance.clear()
                    timerMesure.reset(t)
                    continue
            else:
                thetar = direction_marche
                Vr = v_avance
                WPManager.xr = robot.x + math.cos(thetar)
                WPManager.yr = robot.y + math.sin(thetar)

        elif etat == 'aller_point_detection':
            if not cible_depuis_fusion and point_dans_anciens_cercles(point_cible, cercles_explores):
                point_cible = None
                etat = 'avance'
                cible_depuis_fusion = False
                point_objectif = None
                navigation_intelligente_active = True
                compteur_gradient_nul = 0
                Vr = 0.0
                continue
            thetar = math.atan2(point_cible[1] - robot.y, point_cible[0] - robot.x)
            dist_point = distance_points((robot.x, robot.y), point_cible)
            if dist_point <= tolerance_point:
                point_courant = point_cible
                pot_point_courant = pot_actuel
                if mettre_a_jour_matrice(matrice_mesures, robot.x, robot.y, pot_point_courant):
                    mesures_enregistrees.append((robot.x, robot.y, pot_point_courant))
                etat = 'initialiser_cercle'
                cible_depuis_fusion = False
                point_objectif = None
                compteur_gradient_nul = 0
                Vr = 0.0
                timerMesure.reset(t)
            else:
                Vr = v_point
            WPManager.xr = point_cible[0]
            WPManager.yr = point_cible[1]

        elif etat == 'initialiser_cercle':
            compteur_cercles_consecutifs += 1
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
                if mettre_a_jour_matrice(matrice_mesures, robot.x, robot.y, pot_point):
                    mesures_enregistrees.append((robot.x, robot.y, pot_point))
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
            mesures_actuelles = list(mesures_cercle)
            points_a_explorer = []
            zones_a_ajouter = []
            point_prioritaire = None
            meilleur_pot_prioritaire = None
            limite_cercles_atteinte = compteur_cercles_consecutifs >= MAX_CERCLES_CONSECUTIFS
            for detection in detections:
                mesure_ref = detection['mesure']
                pos_mes = mesure_ref[2]
                dist_mesure = distance_points(centre_cercle, pos_mes)
                if dist_mesure <= dist_fusion:
                    pos_estimee = estimer_centre_ligne(centre_cercle, pot_centre_cercle, mesure_ref)
                    zones_a_ajouter.append(pos_estimee)
                    if len(zones_trouvees) + len(zones_a_ajouter) < nb_zones:
                        if meilleur_pot_prioritaire is None or mesure_ref[0] > meilleur_pot_prioritaire:
                            point_prioritaire = pos_mes
                            meilleur_pot_prioritaire = mesure_ref[0]
                else:
                    points_a_explorer.append(pos_mes)

            if points_a_explorer:
                points_a_explorer = fusionner_positions(points_a_explorer, dist_fusion)
                for pt in points_a_explorer:
                    if point_dans_anciens_cercles(pt, cercles_explores):
                        continue
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

            prochain_point = None
            prioritaire_selectionne = False
            if (
                not limite_cercles_atteinte
                and point_prioritaire is not None
                and not retour_active
            ):
                prochain_point = point_prioritaire
                prioritaire_selectionne = True
            elif not limite_cercles_atteinte and file_points and not retour_active:
                prochain_point = file_points.pop(0)

            if len(zones_trouvees) >= nb_zones and not retour_active:
                retour_active = True
                etat = 'retour'
                compteur_cercles_consecutifs = 0
            elif limite_cercles_atteinte and not retour_active:
                file_points.clear()
                etat = 'aller_point_objectif'
                point_objectif = None
                navigation_intelligente_active = False
                compteur_gradient_nul = 0
                historique_mesures_avance.clear()
                compteur_cercles_consecutifs = 0
            elif prochain_point is not None and not retour_active:
                point_cible = prochain_point
                if prioritaire_selectionne:
                    cible_depuis_fusion = prioritaire_selectionne
                etat = 'aller_point_detection'
            else:
                etat = 'avance'
                navigation_intelligente_active = True
                compteur_gradient_nul = 0
                point_objectif = None

            cercle_actuel = {
                'centre': centre_cercle,
                'rayon': rayon_cercle,
                'points': [mes[2] for mes in mesures_actuelles],
                'mesures': mesures_actuelles
            }
            cercles_explores.append(cercle_actuel)
            nettoyer_file_points(file_points, cercles_explores)

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

#simu.plotXYTheta(2)
#simu.plotVOmega(3)
#simu.plotPotential(4)
#simu.plotPotential3D(5)

reconstruire_carte_emissions(
    matrice_mesures,
    pot_seuil,
    zones_trouvees,
    rayon_detection=rayon_cercle,
    mesures=mesures_enregistrees,
)

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