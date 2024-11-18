import spikeyboi.spikey


def gradient_map(self, color_stops=None):

    if color_stops is None:
        color_stops = [
            (0.0, (255,0,128,200)),
            (0.5, (0,0,0,200)),
            (1.0, (0,255,120,200)),
        ]

    color_stops = sorted(color_stops, key=lambda x: x[0])

    positions = np.array([pos for pos,_ in color_stops])
    colors = np.array([color for _,color in color_stops])

    rgba = np.zeros((*value.shape,4), dtype=float)

    for i in range(len(positions) - 1):
        start_pos = positions[i]
        end_pos = positions[i + 1]
        start_color = colors[i]
        end_color = colors[i + 1]

        segment_mask = (value >= start_pos) & (value <= end_pos)

        if not np.any(segment_mask):
            continue

        segment_values = value[segment_mask]

        t = (segment_values - start_pos) / (end_pos - start_pos)

        rgba[segment_mask] = (start_color[np.newaxis, :] *
                                (1 - t[:, np.newaxis]) +
                                end_color[np.newaxis, :] * t[:, np.newaxis])
    rgba[value < positions[0]] = colors[0]
    rgba[value > positions[-1]] = colors[-1]

    return rgba.astype(int)
def gradient_map(value, color_stops=None):
    """
    Generate a gradient mapping for values between 0 and 1 using arbitrary color stops.
    
    Parameters:
    value: ndarray
        Array of values between 0 and 1 to be mapped to colors
    color_stops: list of tuples
        List of (position, (r,g,b,a)) tuples where position is between 0 and 1
        If None, defaults to three-stop gradient (original behavior)
    
    Returns:
    ndarray: RGBA values for each input value
    """
    if color_stops is None:
        # Default to original three-stop behavior
        color_stops = [
            (0.0, (255, 0, 128, 200)),
            (0.5, (0, 0, 0, 200)),
            (1.0, (0, 255, 120, 200))
        ]
    
    # Ensure color stops are sorted by position
    color_stops = sorted(color_stops, key=lambda x: x[0])
    
    # Convert color values to numpy arrays for easier computation
    positions = np.array([pos for pos, _ in color_stops])
    colors = np.array([color for _, color in color_stops])
    
    # Initialize output array
    rgba = np.zeros((*value.shape, 4), dtype=float)
    
    # For each pair of consecutive stops
    for i in range(len(positions) - 1):
        start_pos = positions[i]
        end_pos = positions[i + 1]
        start_color = colors[i]
        end_color = colors[i + 1]
        
        # Create mask for values in this segment
        segment_mask = (value >= start_pos) & (value <= end_pos)
        
        # Skip if no values in this segment
        if not np.any(segment_mask):
            continue
        
        # Calculate interpolation factor for this segment
        segment_values = value[segment_mask]
        t = (segment_values - start_pos) / (end_pos - start_pos)
        
        # Interpolate colors
        rgba[segment_mask] = (
            start_color[None, :] * (1 - t[:, None]) +
            end_color[None, :] * t[:, None]
        )
    
    # Handle edge cases
    rgba[value < positions[0]] = colors[0]
    rgba[value > positions[-1]] = colors[-1]
    
    return rgba.astype(int)

# color maps dictionary
colormaps = {
    'rdgr': lambda x: gradient_map(x, [
        (0.0, (255, 0, 120, 200)),
        (0.5, (0, 0, 0, 200)),
        (1.0, (0, 255, 120, 200))
    ]),
    'puor': lambda x: gradient_map(x, [
        (0.0, (128, 50, 128, 200)),
        (0.5, (0, 0, 0, 0)),
        (1.0, (247, 247, 247, 200))
    ]),
    'ylgn': lambda x: gradient_map(x, [
        (0.0, (100, 255, 0, 200)),
        (0.5, (0, 0, 0, 200)),
        (1.0, (0, 255, 0, 200))
    ]),
    'rdbu': lambda x: gradient_map(x, [
        (0.0, (255, 0, 50, 200)),
        (0.5, (0, 0, 0, 0)),
        (1.0, (50, 0, 255, 200))
    ]),
    'zebra': lambda x: gradient_map(x, [
        (0.0, (0, 0, 0, 200)),
        (0.5, (100, 100, 100, 200)),
        (1.0, (255, 255, 255, 200))
    ]),
    'mabu': lambda x: gradient_map(x, [
        (0.0, (255, 0, 255, 200)),
        (0.5, (0, 0, 0, 0)),
        (1.0, (0, 0, 255, 200))
    ]),
    'plasma': lambda x: gradient_map(x, [
        (0.0, (13, 8, 135, 200)),
        (0.5, (189, 55, 84, 200)),
        (1.0, (246, 251, 130, 200))
    ]),
    'coolwarm': lambda x: gradient_map(x, [
        (0.0, (59, 76, 192, 200)),
        (0.5, (255, 255, 255, 200)),
        (1.0, (180, 4, 38, 200))
    ]),
    'inferno': lambda x: gradient_map(x, [
        (0.0, (0, 0, 4, 200)),
        (0.5, (120, 15, 100, 200)),
        (1.0, (252, 255, 164, 200))
    ]),
    'magma': lambda x: gradient_map(x, [
        (0.0, (0, 0, 3, 200)),
        (0.5, (128, 18, 97, 200)),
        (1.0, (252, 253, 191, 200))
    ]),
    'viridis': lambda x: gradient_map(x, [
        (0.0, (68, 1, 84, 200)),
        (0.5, (32, 144, 140, 200)),
        (1.0, (253, 231, 37, 200))
    ]),
    'spectral': lambda x: gradient_map(x, [
        (0.0, (158, 1, 66, 200)),
        (0.5, (255, 255, 191, 200)),
        (1.0, (94, 79, 162, 200))
    ]),
    'fallback': lambda x: gradient_map(x)
}
