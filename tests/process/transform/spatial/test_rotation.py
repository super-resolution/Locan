import numpy as np
import pytest

from locan import Rotation2D, Rotation3D


class TestRotation2D:

    def test_init(self):
        # print(help(Rotation2D))
        rotation = Rotation2D.from_angle(angle=np.pi)
        assert rotation._angle_in_radians == pytest.approx(np.pi)
        assert np.allclose(rotation._matrix, [[-1, 0], [0, -1]])
        assert rotation.as_angle() == pytest.approx(np.pi)
        assert rotation.as_angle(degrees=True) == pytest.approx(180)
        assert np.allclose(rotation.matrix, [[-1, 0], [0, -1]])
        assert np.allclose(rotation.as_matrix(), [[-1, 0], [0, -1]])

        rotation = Rotation2D.from_angle(angle=90, degrees=True)
        assert rotation.as_angle() == pytest.approx(np.pi / 2)
        assert rotation.as_angle(degrees=True) == pytest.approx(90)
        assert np.allclose(rotation.as_matrix(), [[0, -1], [1, 0]])

        rotation = Rotation2D.from_matrix(matrix=[[-1, 0], [0, -1]])
        assert rotation.as_angle() == pytest.approx(np.pi)
        assert np.allclose(rotation.as_matrix(), [[-1, 0], [0, -1]])

    def test_align_vectors(self):
        rotation = Rotation2D.align_vectors(vector=[1, 1])
        assert rotation.as_angle(degrees=True) == pytest.approx(45)

        rotation = Rotation2D.align_vectors(vector=[1, 0], other_vector=[1, 1])
        assert rotation.as_angle(degrees=True) == pytest.approx(-45)

        rotation = Rotation2D.align_vectors(vector=[0, 2], other_vector=[1, 1])
        assert rotation.as_angle(degrees=True) == pytest.approx(45)

    def test_apply(self):
        rotation = Rotation2D.from_angle(angle=90, degrees=True)
        new_vector = rotation.apply(vectors=(0, 10))
        assert np.allclose(new_vector, [-10, 0])

        rotation = Rotation2D.from_angle(angle=180, degrees=True)
        new_vector = rotation.apply(vectors=(0, 10))
        assert np.allclose(new_vector, [0, -10])

        rotation = Rotation2D.from_angle(angle=45, degrees=True)
        new_vector = rotation.apply(vectors=(0, 1))
        assert np.allclose(new_vector, [-np.sqrt(2) / 2, np.sqrt(2) / 2])

        rotation = Rotation2D.from_angle(angle=90, degrees=True)
        new_vectors = rotation.apply(vectors=((0, 1), (0, 10)))
        assert np.allclose(new_vectors, [[-1, 0], [-10, 0]])

    def test_invert(self):
        rotation = Rotation2D.from_angle(angle=90, degrees=True)
        new_rotation = rotation.inv()
        assert new_rotation.as_angle() == pytest.approx((-1) * rotation.as_angle())

        for angle in np.linspace(0, 2 * np.pi, 10):
            converted_angle = angle if angle < np.pi else (angle - 2 * np.pi)
            new_rotation = Rotation2D.from_angle(angle=angle).inv()
            assert new_rotation.as_angle() == pytest.approx(-converted_angle)


class TestRotation3D:

    def test_init(self):
        # print(help(Rotation3D))
        matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        rotation = Rotation3D.from_matrix(matrix=matrix)
        assert np.allclose(rotation.as_matrix(), matrix)
        new_vector = rotation.apply((1, 2, 3))
        assert np.allclose(new_vector, [-2.0, 1.0, 3.0])
