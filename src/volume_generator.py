import glfw
from tqdm import tqdm
import cv2
import os
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math
import pyrr
import openmesh as om
import json
from PIL import Image  # , ImageDraw, ImageOps
from geometry_generator import GeometryGenerator
from color_maps import ColorMaps
from omegaconf import DictConfig


class VolumeGenerator:

    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.width = self.height = self.cfg.rendering.resolution
        self.src_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(self.src_dir, "..", "data"))
        self.geometry_generator = GeometryGenerator(cfg=self.cfg.geometry)
        self.color_map = ColorMaps(self.cfg.texture)
        if self.cfg.random:
            self.geometry_dir = os.path.join(self.data_dir, "tree_geo_random")
            self.wood_color_map_dir = os.path.join(
                self.data_dir, "wood_color_maps_random"
            )
        else:
            self.geometry_dir = os.path.join(self.data_dir, "tree_geo_original")
            self.wood_color_map_dir = os.path.join(
                self.data_dir, "wood_color_maps_original"
            )

    def load_texture(self, i, path, nearest=False, repeat_x_edge=False):
        gltex = [
            GL_TEXTURE0,
            GL_TEXTURE1,
            GL_TEXTURE2,
            GL_TEXTURE3,
            GL_TEXTURE4,
            GL_TEXTURE5,
            GL_TEXTURE6,
            GL_TEXTURE7,
        ]
        image = Image.open(path)
        width = int(image.size[0])
        height = int(image.size[1])
        number_of_channels = 3
        tex = np.array(image.getdata()).reshape(width, height, number_of_channels)
        tex = np.array(list(tex), np.uint8)
        glActiveTexture(gltex[i])
        tex_handle = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_handle)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            width,
            height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            tex,
        )
        if nearest:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        if repeat_x_edge:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def generate_volume(self) -> np.ndarray:

        end = 0.45
        if self.cfg.random:
            image, end = self.color_map.generate_color_map()
            # save color map
            color_map_path = os.path.join(self.wood_color_map_dir, "wood_bar_color.bmp")
            os.makedirs(self.wood_color_map_dir, exist_ok=True)
            image.save(color_map_path)

        # # new log geometry
        self.geometry_generator.generate_geometry(end)

        ### WINDOW SETUP ###########################################################

        if not glfw.init():
            return

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            self.width, self.height, "Procedural Knots", None, None
        )

        if not self.window:
            glfw.terminate()
            return

        glfw.make_context_current(self.window)

        ### LOAD INPUT 3D MODEL ####################################################

        # Paths
        mesh_path = os.path.join(self.data_dir, "3d_model", "plank.obj")
        mesh = om.read_trimesh(mesh_path, vertex_normal=True)

        # Vertices with normals
        point_array = mesh.points()
        normal_array = mesh.vertex_normals()
        vertex_array = np.concatenate(
            (point_array, normal_array), axis=1
        )  # [x,y,z] + [nx,ny,nz] --> [x,y,z,nx,ny,nz]
        self.verts = np.array(vertex_array.flatten(), dtype=np.float32)

        # Face indices
        face_array = mesh.face_vertex_indices()
        self.indices = np.array(face_array.flatten(), dtype=np.uint32)

        ### LOAD VERTEX AND FRAGEMENT SHADERS FROM EXTERNAL FILES ##################

        # Buffer vertices and indices
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4 * len(self.verts), self.verts, GL_DYNAMIC_DRAW)

        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            4 * len(self.indices),
            self.indices,
            GL_DYNAMIC_DRAW,
        )

        # Vertex attribute pointers
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0)
        )  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12)
        )  # normal
        glEnableVertexAttribArray(1)

        VERTEX_SHADER = open(os.path.join(self.src_dir, "main.vert"), "r").read()
        FRAGMENT_SHADER = open(os.path.join(self.src_dir, "main.frag"), "r").read()

        # Compile The Program and shaders
        shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
        )

        ### LOAD TEXTURE MAPS ######################################################

        glEnable(GL_TEXTURE_2D)

        # Wood colors etc.
        self.load_texture(
            0, os.path.join(self.wood_color_map_dir, "wood_bar_color.bmp")
        )

        # Internal tree log skeleton geometry
        self.load_texture(
            3,
            os.path.join(self.geometry_dir, "pith_and_radius_map.bmp"),
            repeat_x_edge=True,
        )
        self.load_texture(4, os.path.join(self.geometry_dir, "knot_height_map.bmp"))
        self.load_texture(
            5, os.path.join(self.geometry_dir, "knot_orientation_map.bmp")
        )
        self.load_texture(
            6,
            os.path.join(self.geometry_dir, "knot_state_map.bmp"),
            nearest=True,
        )

        glUseProgram(shader)

        texLocCol = glGetUniformLocation(shader, "ColorMap")
        glUniform1i(texLocCol, 0)
        texLocSpec = glGetUniformLocation(shader, "SpecularMap")
        glUniform1i(texLocSpec, 1)
        texLocNorm = glGetUniformLocation(shader, "NormalMap")
        glUniform1i(texLocNorm, 2)
        texLocNorm = glGetUniformLocation(shader, "PithRadiusMap")
        glUniform1i(texLocNorm, 3)
        texLocNorm = glGetUniformLocation(shader, "KnotHeightMap")
        glUniform1i(texLocNorm, 4)
        texLocNorm = glGetUniformLocation(shader, "KnotOrientMap")
        glUniform1i(texLocNorm, 5)
        texLocNorm = glGetUniformLocation(shader, "KnotStateMap")
        glUniform1i(texLocNorm, 6)
        endLoc = glGetUniformLocation(shader, "end")
        glUniform1f(endLoc, end)

        # knot color
        knot_color = np.random.uniform(
            self.cfg.texture.knot_color_range[0], self.cfg.texture.knot_color_range[1]
        )
        knotColor = glGetUniformLocation(shader, "knotColor")
        glUniform1f(knotColor, knot_color)

        ### SET SHADER PARAMETERS ##################################################

        # Set tree log properties
        f = open(os.path.join(self.geometry_dir, "map_params.json"))
        rmin, rmax, hmax, knum = json.load(f)
        endtime = hmax / rmin / 0.25
        f.close()
        # print(rmin, rmax, hmax, knum)

        rminLoc = glGetUniformLocation(shader, "rmin")
        glUniform1f(rminLoc, rmin)
        rmaxLoc = glGetUniformLocation(shader, "rmax")
        glUniform1f(rmaxLoc, rmax)
        hmaxLoc = glGetUniformLocation(shader, "hmax")
        glUniform1f(hmaxLoc, hmax)
        knumLoc = glGetUniformLocation(shader, "N")
        glUniform1i(knumLoc, knum)

        # Model matrix
        mrot = 0.0
        model = np.array(pyrr.Matrix44.from_z_rotation(mrot))
        modelLoc = glGetUniformLocation(shader, "model")
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model)

        # View matrix
        rot_x = pyrr.Matrix44.from_x_rotation(0.00)
        rot_z = pyrr.Matrix44.from_z_rotation(0.0)
        view = np.array(rot_x * rot_z)
        # change position of view
        view[3][1] += 0.182  # fix offset INVESTIGATE

        viewLoc = glGetUniformLocation(shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view)

        # Seed
        seed = 42
        seed_loc = glGetUniformLocation(shader, "randomSeed")
        glUniform1i(seed_loc, seed)

        ### DRAW ###################################################################

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        volume = []
        # volume = np.zeros((self.height, self.width, 1, self.n_slices), dtype=np.uint8)
        # while not glfw.window_should_close(self.window):
        n_slices = int(self.cfg.geometry.log_length / self.cfg.geometry.slice_size)
        for time_idx in tqdm(
            range(0, n_slices),
            desc="Generating volume slices",
        ):

            # interpolate time_idx from z to endtime
            time = time_idx / n_slices * endtime

            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Pass time variable to fragment shader (for animation)
            timeLoc = glGetUniformLocation(shader, "time")
            # glUniform1f(timeLoc, glfw.get_time())
            glUniform1f(timeLoc, time)

            # Draw mesh
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

            # Swap buffers
            glfw.swap_buffers(self.window)

            # ---------- READ PIXELS ----------
            buffer = glReadPixels(
                0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE
            )

            #       Convert to NumPy (flip vertically because OpenGL origin = bottom-left)
            img = np.frombuffer(buffer, dtype=np.uint8).reshape(
                self.height, self.width, 3
            )

            img = np.flipud(img)

            volume.append(img)
            # volume[:, :, 0, time_idx] = img[:, :, 0]

            if self.cfg.show:
                cv2.imshow("Rendered Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        # import ipdb
        #
        # ipdb.set_trace()
        volume = np.stack(volume, axis=-1)
        cv2.destroyAllWindows()
        glfw.terminate()

        # permute last with first axis
        volume = np.transpose(volume, (3, 0, 1, 2)).squeeze()

        print("Postprocessing for offset")
        # volume postprocessing
        offset = np.where(volume > 0)[1].min()
        microtec_offset = int(
            np.random.uniform(*self.cfg.postprocessing.microtec_offset)
        )
        offset_delta = offset - microtec_offset
        # slide the volume by offset delta
        volume = np.roll(volume, -offset_delta, axis=1)

        return volume


if __name__ == "__main__":
    generator = VolumeGenerator()
    generator.generate_volume()
