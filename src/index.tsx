import type { Frame } from "react-native-vision-camera";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedLabels {
  scores: number[];
  boundingBoxes: BoundingBox[];
  detectionCount: number[];
  categories: number[];
}

/**
 * Wraps the globally exported function
 * `frame` - the frame to detect the labels from
 * `modelPath` - the file path to the model you want to use for inference
 */
export function detectLabel(frame: Frame, modelPath: string): DetectedLabels | undefined {
  "worklet";
  // @ts-ignore
  // eslint-disable-next-line no-undef
  return __detectLabel(frame, modelPath);
}