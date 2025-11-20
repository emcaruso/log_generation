import numpy as np
import joblib
import math
import os
import cv2
from PIL import Image, ImageFilter
import json
from perlin_numpy import generate_perlin_noise_2d
from random_walk import RandomWalkGenerator, plot_trajectory


def flatten(t):
    return [item for sublist in t for item in sublist]


def mapFromTo(x, a, b, c, d):
    y = (x - a) / (b - a) * (d - c) + c
    return y


def smoothen_rads(R, n0, n1):
    R_new = np.copy(R)
    for i in range(R.shape[0]):
        for j in range(360):
            near_rads = []
            for a in range(-n0, n0 + 1):
                for b in range(-n1, n1 + 1):
                    ii = (i + a) % R.shape[0]
                    jj = (j + b) % 360
                    near_rads.append(R[ii][jj])
            rad = sum(near_rads) / len(near_rads)
            R_new[i][j] = rad
    return R_new


def get_min_max_radii_and_tree_height(path):
    f = open(path)
    rads = json.load(f)
    rads_flat = flatten(rads)
    f.close()
    return min(rads_flat), max(rads_flat), len(rads) * 10


def load_and_save_pith_and_outer_shape(
    path_pith, path_rad, filename, min_r, max_r, show=True
):

    # pith
    f = open(path_pith)
    piths = json.load(f)
    znum = len(piths)
    pith_and_rads = np.zeros((znum, 360, 3), dtype="float64")
    mid_index = int(0.5 * len(piths))
    x0 = piths[mid_index][0]
    y0 = piths[mid_index][1]

    for i in range(znum):
        x = piths[i][0] - x0
        y = piths[i][1] - y0
        x = mapFromTo(x, -0.5 * min_r, 0.5 * min_r, 0, 255)
        y = mapFromTo(y, -0.5 * min_r, 0.5 * min_r, 0, 255)
        for j in range(360):
            pith_and_rads[i][j][0] = x  # red
            pith_and_rads[i][j][1] = y  # green
    f.close()

    # outer shape rads
    f = open(path_rad)
    rads = json.load(f)
    rads = np.array(rads)
    for i in range(znum):
        for j in range(360):
            rads[i][j] = mapFromTo(rads[i][j] - min_r, 0, max_r - min_r, 0, 255)
    rads = smoothen_rads(rads, 5, 2)
    for i in range(znum):
        for j in range(360):
            pith_and_rads[i][j][2] = rads[i][j]  # blue
    f.close()

    # image
    img = Image.fromarray(np.uint8(pith_and_rads), "RGB")
    img = img.resize((180, znum))
    img.save(filename)
    if show:
        img.show()


def load_and_save_knot_params_to_hmap_rmap(
    path,
    filename_hmap,
    filename_rmap,
    filename_smap,
    min_r,
    max_r,
    max_h,
    col_width=128,
    row_height=1,
):

    f = open(path)
    knot_params = json.load(f)
    f.close()
    k_num = len(knot_params)

    width = col_width
    # in pixels
    knot_height = np.zeros((row_height * k_num, width, 3))
    knot_rotation = np.zeros((row_height * k_num, width, 3))
    knot_state = np.zeros((row_height * k_num, width, 3))

    for i in range(k_num):
        A, B, C, D, E, F, G, H, I = knot_params[i]

        if I < 0.1:
            continue  # Dies super early

        H_mapped = mapFromTo(H, 0, max_r, 0, width)  # end distance
        I_mapped = mapFromTo(I, 0, max_r, 0, width)  # death distance

        for j in range(width):
            rp = mapFromTo(j, 0, width, 0, max_r)  # distnace from pith

            # height
            z_start = C
            z_fine = D * math.sqrt(rp) + E * rp
            z_fine_up = 0.0
            z_fine_dn = 0.0
            if z_fine > 0:
                z_fine_up = z_fine
            else:
                z_fine_dn = -z_fine

            # rotation
            om_start = (F + 2 * math.pi) % 2 * math.pi
            if j == 0:
                om_twist = 0.0
            else:
                om_twist = G * math.log(rp)
            om_twist_ccw = 0.0
            om_twist_cw = 0.0
            if om_twist > 0:
                om_twist_ccw = om_twist
            else:
                om_twist_cw = -om_twist

            for k in range(row_height):

                ## height
                knot_height[i * row_height + k][j][0] = mapFromTo(
                    z_start, 0, max_h, 0, 255
                )  # red   = start
                knot_height[i * row_height + k][j][1] = mapFromTo(
                    z_fine_up, 0.0, min_r, 0, 255
                )  # green = up
                knot_height[i * row_height + k][j][2] = mapFromTo(
                    z_fine_dn, 0.0, min_r, 0, 255
                )  # blue  = down

                ## rotation
                knot_rotation[i * row_height + k][j][0] = mapFromTo(
                    om_start, 0, 2.0 * math.pi, 0, 255
                )  # red   = start
                knot_rotation[i * row_height + k][j][1] = mapFromTo(
                    om_twist_ccw, 0, 0.5 * math.pi, 0, 255
                )  # green = left twist
                knot_rotation[i * row_height + k][j][2] = mapFromTo(
                    om_twist_cw, 0, 0.5 * math.pi, 0, 255
                )  # blue  = right twist

                ## state
                if j < I_mapped:  # alive
                    knot_state[i * row_height + k][j][0] = 255
                    # red   = alive
                knot_state[i * row_height + k][j][1] = mapFromTo(
                    I, 0.0, max_r, 0, 255
                )  # green = dead
                # knot_state[i*row_height+k][j][2] = mapFromTo(H,0.0,max_r,0,255)    #blue  = broken off

    img = Image.fromarray(np.uint8(knot_height), "RGB")
    img.save(filename_hmap)
    # img.show()

    img = Image.fromarray(np.uint8(knot_rotation), "RGB")
    img.save(filename_rmap)
    # img.show()

    img = Image.fromarray(np.uint8(knot_state), "RGB")
    img.save(filename_smap)
    # img.show()

    return k_num


def closest_divisor(A, B):
    divisors = []

    # find divisors up to sqrt(A)
    for i in range(1, int(A**0.5) + 1):
        if A % i == 0:
            divisors.append(i)
            if i != A // i:
                divisors.append(A // i)

    # return divisor closest to B
    return min(divisors, key=lambda d: abs(d - B))


def farthest_point_sampling(points, k):
    """
    points: (N, D) array
    k: number of knots to select
    """
    points = np.asarray(points)
    N = len(points)

    # pick a random starting point
    idx = np.random.randint(N)
    selected = [idx]

    # compute squared distances from that point
    dist = np.sum((points - points[idx]) ** 2, axis=1)

    for _ in range(1, k):
        # pick the farthest point
        idx = np.argmax(dist)
        selected.append(idx)

        # update min distance to the set of selected knots
        dist = np.minimum(dist, np.sum((points - points[idx]) ** 2, axis=1))

    return np.array(selected)


class GeometryGenerator:

    def __init__(
        self,
        random: bool = False,
        radius_min: int = 104,
        radius_max: int = 143,
        n_slices: int = 300,
        n_knots: int = 200,
        perlin_density: int = 5,
        gmm_model: str = "aic",  # "bic", "aic" or "single"
        minimal_save: bool = False,
    ):
        self.random = random
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.n_slices = n_slices
        self.n_knots = n_knots
        self.perlin_density = perlin_density
        self.minimal_save = minimal_save
        self.gmm_model = gmm_model
        self.src_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(self.src_dir, "..", "data"))
        self.configs_dir = os.path.abspath(os.path.join(self.src_dir, "..", "configs"))
        self.statistics_dir = os.path.abspath(
            os.path.join(self.data_dir, "tree_statistics")
        )
        if self.random:
            ext = "random"
        else:
            ext = "original"
        self.tree_geo_dir = os.path.abspath(
            os.path.join(self.data_dir, f"tree_geo_{ext}")
        )

    def generate_geo_metadata(self):

        ####### RADII GENERATION ##############
        # for radii, generate perlin noise
        ratio = int(self.n_slices / 360) + 1
        res_x = 360
        res_y = res_x * ratio
        # noise = generate_perlin_noise_2d((res_y, res_x), (mul, mul ), (False, True))
        # multiscale noise  HARDCODED !!!
        noise = np.zeros((res_y, res_x))
        coeffs = [1, 2, 4, 6, 8]
        for i in coeffs:
            rep_x = closest_divisor(res_x, i)
            rep_y = closest_divisor(res_y, i)
            noise += (
                generate_perlin_noise_2d((res_y, res_x), (rep_y, rep_x), (False, True))
                * 0.06
                * ((coeffs[-1] + 1) - i)
            )
        # crop noise
        noise = noise[0 : self.n_slices, 0:res_x]

        # # normalize noise
        # noise = (noise - min) / (max - min)

        if not self.minimal_save:
            # save img
            Image.fromarray(
                np.uint8((noise - noise.min() / (noise.max() - noise.min())) * 255)
            ).save(os.path.join(self.tree_geo_dir, "perlin_noise.bmp"))

        # save radii json ( noise as list of lists [n_slices, 360])
        radii = noise * (self.radius_max - self.radius_min) + self.radius_min
        radii_list = radii.tolist()
        with open(os.path.join(self.tree_geo_dir, "geo_radii.json"), "w") as f:
            json.dump(radii_list, f)

        ####### KNOTS GENERATION ##############

        knots_list = []
        for i in range(0, 9):
            # load gmm model
            gmm_path = os.path.join(
                self.statistics_dir, "gmm_models", f"{self.gmm_model}_dim{i+1}.pkl"
            )
            with open(gmm_path, "rb") as f:
                gmm = joblib.load(f)
                # sample n_knots values
                values = gmm.sample(self.n_knots * 20)[0].squeeze()
                knots_list.append(values.tolist())

        samples = np.array(knots_list).T
        selected_indices = farthest_point_sampling(samples, self.n_knots)
        knots_list = samples[selected_indices].T
        knots_list = knots_list.tolist()

        # save knots json ( list of knots with 9 parameters)
        knots_params = list(zip(*knots_list))
        with open(os.path.join(self.tree_geo_dir, "geo_knots.json"), "w") as f:
            json.dump(knots_params, f)

        ####### PITH GENERATION ##############

        # load statistics
        stats_path = os.path.join(self.statistics_dir, "tree_data_statistics.json")
        stats = json.load(open(stats_path))
        min_p = stats["geo_pith"]["min"]
        max_p = stats["geo_pith"]["max"]
        n_steps = self.n_slices
        mean_p = stats["geo_pith"]["mean"]
        std_p = stats["geo_pith"]["std"]
        p0_x = np.random.normal(mean_p[0], std_p[0])
        p0_y = np.random.normal(mean_p[1], std_p[1])
        p0 = np.array([p0_x, p0_y])

        self.random_walk_generator = RandomWalkGenerator(n_steps, min_p, max_p, p0)
        pith_trajectory = self.random_walk_generator.get_trajectory()

        # save trajectory image
        if not self.minimal_save:
            plot_trajectory(
                pith_trajectory, os.path.join(self.tree_geo_dir, "pith_trajectory.png")
            )

        z = np.array([i for i in range(0, self.n_slices * 10, 10)])[..., None]
        traj = np.concatenate([pith_trajectory, z], axis=1)

        # save json (list of lists [n_slices, 3])
        with open(os.path.join(self.tree_geo_dir, "geo_pith.json"), "w") as f:
            json.dump(traj.tolist(), f)

    def generate_geometry(self):

        if self.random:
            self.generate_geo_metadata()

        # generated geometry files
        pith_path = os.path.join(self.tree_geo_dir, "geo_pith.json")
        radii_path = os.path.join(self.tree_geo_dir, "geo_radii.json")
        knot_path = os.path.join(self.tree_geo_dir, "geo_knots.json")
        fname_pmap = os.path.join(self.tree_geo_dir, "pith_and_radius_map.bmp")
        fname_hmap = os.path.join(self.tree_geo_dir, "knot_height_map.bmp")
        fname_rmap = os.path.join(self.tree_geo_dir, "knot_orientation_map.bmp")
        fname_smap = os.path.join(self.tree_geo_dir, "knot_state_map.bmp")
        fname_parms = os.path.join(self.tree_geo_dir, "map_params.json")

        min_rad, max_rad, max_height = get_min_max_radii_and_tree_height(radii_path)

        print("Min radius:", min_rad, "mm")
        print("Max radius:", max_rad, "mm")
        print("Tree height", max_height, "mm")

        load_and_save_pith_and_outer_shape(
            pith_path, radii_path, fname_pmap, min_rad, max_rad, show=False
        )

        num_knots = load_and_save_knot_params_to_hmap_rmap(
            knot_path,
            fname_hmap,
            fname_rmap,
            fname_smap,
            min_rad,
            max_rad,
            max_height,
            col_width=32,
            row_height=4,
        )

        print("Number of knots", num_knots)

        with open(fname_parms, "w") as f:
            json.dump([min_rad, max_rad, max_height, num_knots], f)


if __name__ == "__main__":
    geometry_generator = GeometryGenerator(random=True)
    geometry_generator.generate_geometry()
