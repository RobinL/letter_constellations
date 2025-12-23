export async function loadImageBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load image: ${url}`);
  }
  const blob = await response.blob();
  return createImageBitmap(blob);
}
