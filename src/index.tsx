import type { VisionCameraProxy, Frame } from "react-native-vision-camera";

const plugin = VisionCameraProxy.getFrameProcessorPlugin('detectLabel')

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
  if (plugin == null) throw new Error('Failed to load Frame Processor Plugin "detectLabel"!')
  // @ts-ignore
  // eslint-disable-next-line no-undef
  return plugin.call(frame, {modelPath});
}