import type { MetadataRoute } from 'next';
export default function manifest(): MetadataRoute.Manifest {
  return { id: '/', name: 'Waypoint', short_name: 'Waypoint', description: 'Camera view and spoken hazard warnings.',
    start_url: '/', scope: '/', display: 'standalone', orientation: 'portrait', background_color: '#f7f7f8', theme_color: '#f7f7f8',
    icons: [{ src: '/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' }, { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' }, { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' }] };
}
