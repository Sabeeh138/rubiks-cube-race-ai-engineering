import assert from "node:assert/strict";
import test from "node:test";

import { FACE_ORDER, faceletsToFaces, isSolvedFacelets } from "../../app/static/cube.mjs";

const solved = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB";
const afterR = "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB";

test("renders solved URFDLB facelets as six labeled 3x3 faces", () => {
  const faces = faceletsToFaces(solved);

  assert.deepEqual(Object.keys(faces), FACE_ORDER);
  assert.ok(FACE_ORDER.every((face) => faces[face].length === 9));
  assert.ok(FACE_ORDER.every((face) => faces[face].every((sticker) => sticker === face)));
  assert.equal(isSolvedFacelets(solved), true);
});

test("renders a backend-generated scrambled facelet string without solving it", () => {
  const faces = faceletsToFaces(afterR);

  assert.ok(FACE_ORDER.every((face) => faces[face].length === 9));
  assert.equal(faces.U[2], "F");
  assert.equal(faces.R[4], "R");
  assert.equal(isSolvedFacelets(afterR), false);
});

test("rejects malformed cube strings", () => {
  assert.throws(() => faceletsToFaces("short"), /54-character/);
  assert.throws(() => faceletsToFaces("X".repeat(54)), /outside URFDLB/);
  assert.throws(() => faceletsToFaces("U".repeat(54)), /nine stickers/);
});
