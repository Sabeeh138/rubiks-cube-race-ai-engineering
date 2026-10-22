export const FACE_ORDER = Object.freeze(["U", "R", "F", "D", "L", "B"]);

/**
 * Split the engine's canonical 54-facelet URFDLB string into six labeled faces.
 * This module only reshapes display data; cube mechanics stay in the backend.
 */
export function faceletsToFaces(facelets) {
  if (typeof facelets !== "string" || facelets.length !== 54) {
    throw new TypeError("Cube state must be a 54-character facelet string.");
  }
  if ([...facelets].some((sticker) => !FACE_ORDER.includes(sticker))) {
    throw new TypeError("Cube state contains a facelet outside URFDLB.");
  }

  const counts = Object.fromEntries(FACE_ORDER.map((face) => [face, 0]));
  for (const sticker of facelets) counts[sticker] += 1;
  if (FACE_ORDER.some((face) => counts[face] !== 9)) {
    throw new TypeError("Cube state must contain nine stickers for each face color.");
  }

  return Object.freeze(
    Object.fromEntries(
      FACE_ORDER.map((face, faceIndex) => [
        face,
        Object.freeze([...facelets.slice(faceIndex * 9, faceIndex * 9 + 9)]),
      ]),
    ),
  );
}

export function isSolvedFacelets(facelets) {
  const faces = faceletsToFaces(facelets);
  return FACE_ORDER.every((face) => faces[face].every((sticker) => sticker === face));
}
