import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Mesh3DResult } from '../types/analysis';
import { 
  Download, 
  RotateCcw, 
  Box, 
  Eye, 
  AlertCircle, 
  Crosshair, 
  CheckCircle2, 
  Compass, 
  Layers, 
  ShieldCheck,
  MapPin
} from 'lucide-react';
import { getMeshDownloadUrl } from '../services/api';

interface Brain3DViewerProps {
  meshData: Mesh3DResult;
  resultId: string;
}

export const Brain3DViewer: React.FC<Brain3DViewerProps> = ({ meshData, resultId }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [showBrainReference, setShowBrainReference] = useState<boolean>(true);
  const [showSlicePlane, setShowSlicePlane] = useState<boolean>(true);
  const [showBeacon, setShowBeacon] = useState<boolean>(true);
  
  const controlsRef = useRef<any>(null);
  const tumorMeshRef = useRef<THREE.Mesh | null>(null);
  const brainMeshRef = useRef<THREE.Mesh | null>(null);
  const slicePlaneRef = useRef<THREE.Mesh | null>(null);
  const beaconGroupRef = useRef<THREE.Group | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 420;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(12, 10, 16);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.replaceChildren(renderer.domElement);

    // 2. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.8);
    dirLight1.position.set(15, 20, 15);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf43f5e, 1.2);
    dirLight2.position.set(-15, -10, -15);
    scene.add(dirLight2);

    const dirLight3 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight3.position.set(0, -15, 0);
    scene.add(dirLight3);

    // 3. Grid Floor
    const grid = new THREE.GridHelper(24, 24, 0x1e293b, 0x0f172a);
    grid.position.y = -7.5;
    scene.add(grid);

    // 4. Brain Anatomical Translucent Surface Reference
    // Standard anatomical orientation:
    // X: Left (-X) to Right (+X) [radius 6.2]
    // Y: Inferior (-Y) to Superior (+Y) [radius 7.44]
    // Z: Posterior (-Z) to Anterior (+Z) [radius 8.37]
    const brainGeo = new THREE.SphereGeometry(6.2, 36, 28);
    brainGeo.scale(1.0, 1.2, 1.35); // Ellipsoidal brain shape
    const brainMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.16,
      roughness: 0.25,
      metalness: 0.1,
      transmission: 0.45,
      wireframe: false,
      depthWrite: false
    });
    const brainMesh = new THREE.Mesh(brainGeo, brainMat);
    scene.add(brainMesh);
    brainMeshRef.current = brainMesh;

    // 4b. Anatomical Orientation Markers (3D Badges)
    const createMarker = (text: string, pos: [number, number, number], color: string) => {
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.beginPath();
        ctx.arc(64, 64, 50, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 4;
        ctx.strokeStyle = color;
        ctx.stroke();

        ctx.font = 'bold 44px sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 64);
      }
      const texture = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.85 });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.set(...pos);
      sprite.scale.set(1.4, 1.4, 1);
      return sprite;
    };

    // Anterior (A), Posterior (P), Left (L), Right (R)
    scene.add(createMarker('A', [0, 0, 9.6], '#38bdf8')); // Front / Anterior
    scene.add(createMarker('P', [0, 0, -9.6], '#94a3b8')); // Back / Posterior
    scene.add(createMarker('L', [-7.4, 0, 0], '#94a3b8')); // Left
    scene.add(createMarker('R', [7.4, 0, 0], '#94a3b8')); // Right

    // 5. Construct Tumor Polygonal Mesh & Features
    if (meshData.has_mesh && meshData.vertices.length > 0 && meshData.faces.length > 0) {
      const geometry = new THREE.BufferGeometry();

      const positions: number[] = [];
      const indices: number[] = [];

      meshData.faces.forEach(face => {
        indices.push(face[0], face[1], face[2]);
      });

      meshData.vertices.forEach(v => {
        positions.push(v[0], v[1], v[2]);
      });

      geometry.setIndex(indices);
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.computeVertexNormals();

      const tumorMat = new THREE.MeshStandardMaterial({
        color: 0xef4444, // Glowing crimson
        roughness: 0.25,
        metalness: 0.25,
        emissive: 0x991b1b,
        emissiveIntensity: 0.55,
        wireframe: wireframe
      });

      const tumorMesh = new THREE.Mesh(geometry, tumorMat);
      scene.add(tumorMesh);
      tumorMeshRef.current = tumorMesh;

      // 5b. Centroid Target Beacon & Target Pin
      const c = meshData.centroid_3d || [0, 0, 0];
      const beaconGroup = new THREE.Group();
      beaconGroup.position.set(c[0], c[1], c[2]);

      // Small glowing center bead
      const beadGeo = new THREE.SphereGeometry(0.28, 16, 16);
      const beadMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
      const beadMesh = new THREE.Mesh(beadGeo, beadMat);
      beaconGroup.add(beadMesh);

      // Targeting Reticle Ring
      const ringGeo = new THREE.RingGeometry(0.65, 0.85, 32);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0xfacc15, side: THREE.DoubleSide, transparent: true, opacity: 0.75 });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = Math.PI / 2;
      beaconGroup.add(ringMesh);

      scene.add(beaconGroup);
      beaconGroupRef.current = beaconGroup;

      // 5c. Translucent Axial Cutting Reference Plane
      const slicePlaneGeo = new THREE.PlaneGeometry(14.5, 17.5);
      const slicePlaneMat = new THREE.MeshBasicMaterial({
        color: 0x06b6d4,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false
      });
      const slicePlane = new THREE.Mesh(slicePlaneGeo, slicePlaneMat);
      slicePlane.rotation.x = Math.PI / 2;
      slicePlane.position.y = c[1];
      scene.add(slicePlane);
      slicePlaneRef.current = slicePlane;
    }

    // 6. Mouse Drag Orbit Navigation
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let spherical = { radius: 22, theta: 0.7, phi: 1.1 };

    const updateCamera = () => {
      camera.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
      camera.position.y = spherical.radius * Math.cos(spherical.phi);
      camera.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
      camera.lookAt(0, 0, 0);
    };
    updateCamera();

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      prevMouse = { x: e.clientX, y: e.clientY };

      spherical.theta -= dx * 0.008;
      spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi - dy * 0.008));
      updateCamera();
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      spherical.radius = Math.max(8, Math.min(45, spherical.radius + e.deltaY * 0.02));
      updateCamera();
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.addEventListener('wheel', onWheel, { passive: false });

    // Preset camera views helper
    controlsRef.current = {
      setView: (theta: number, phi: number, radius = 22) => {
        spherical.theta = theta;
        spherical.phi = phi;
        spherical.radius = radius;
        updateCamera();
      },
      focusCentroid: () => {
        if (!meshData.centroid_3d) return;
        const [cx, cy, cz] = meshData.centroid_3d;
        const theta = Math.atan2(cx, cz);
        const horizDist = Math.sqrt(cx * cx + cz * cz);
        const phi = Math.atan2(horizDist, cy);
        spherical.theta = theta;
        spherical.phi = Math.max(0.3, Math.min(Math.PI - 0.3, phi));
        spherical.radius = 16;
        updateCamera();
      }
    };

    // 7. Animation Loop
    let animId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (autoRotate && !isDragging) {
        spherical.theta += 0.004;
        updateCamera();
      }

      // Gentle pulsing effect for lesion beacon
      if (beaconGroupRef.current) {
        const time = clock.getElapsedTime();
        const scale = 1.0 + Math.sin(time * 3.5) * 0.15;
        beaconGroupRef.current.scale.set(scale, scale, scale);
      }

      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight || 420;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      dom.removeEventListener('wheel', onWheel);
      renderer.dispose();
    };
  }, [meshData]);

  // Wireframe sync
  useEffect(() => {
    if (tumorMeshRef.current) {
      (tumorMeshRef.current.material as THREE.MeshStandardMaterial).wireframe = wireframe;
    }
  }, [wireframe]);

  // Brain reference sync
  useEffect(() => {
    if (brainMeshRef.current) {
      brainMeshRef.current.visible = showBrainReference;
    }
  }, [showBrainReference]);

  // Slice plane sync
  useEffect(() => {
    if (slicePlaneRef.current) {
      slicePlaneRef.current.visible = showSlicePlane;
    }
  }, [showSlicePlane]);

  // Beacon sync
  useEffect(() => {
    if (beaconGroupRef.current) {
      beaconGroupRef.current.visible = showBeacon;
    }
  }, [showBeacon]);

  const setPresetView = (type: 'iso' | 'axial' | 'sagittal' | 'coronal') => {
    if (!controlsRef.current) return;
    setAutoRotate(false);
    switch (type) {
      case 'iso':
        controlsRef.current.setView(0.7, 1.1);
        break;
      case 'axial': // Top (Looking down at axial slice)
        controlsRef.current.setView(0, 0.08);
        break;
      case 'sagittal': // Side (Right to left)
        controlsRef.current.setView(Math.PI / 2, Math.PI / 2);
        break;
      case 'coronal': // Front (Anterior looking posterior)
        controlsRef.current.setView(0, Math.PI / 2);
        break;
    }
  };

  const handleFocusTumour = () => {
    if (!controlsRef.current) return;
    setAutoRotate(false);
    controlsRef.current.focusCentroid();
  };

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-slate-800">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <Box className="h-5 w-5 text-cyan-400" />
            <h3 className="font-semibold text-white text-base">3D Anatomically Co-Registered Visual</h3>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 font-mono">
              Accurate Marching Cubes
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time WebGL polygon surface extraction registered directly to scan coordinates and cranial lobes.
          </p>
        </div>

        {/* Export Buttons */}
        {meshData.has_mesh && (
          <div className="flex items-center space-x-2">
            <a
              href={getMeshDownloadUrl(resultId, 'stl')}
              download
              className="flex items-center space-x-1.5 px-2.5 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
              title="Download binary/ASCII STL 3D Mesh"
            >
              <Download className="h-3.5 w-3.5 text-cyan-400" />
              <span>STL</span>
            </a>
            <a
              href={getMeshDownloadUrl(resultId, 'obj')}
              download
              className="flex items-center space-x-1.5 px-2.5 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
              title="Download OBJ Wavefront 3D Mesh"
            >
              <Download className="h-3.5 w-3.5 text-cyan-400" />
              <span>OBJ</span>
            </a>
          </div>
        )}
      </div>

      {/* Anatomical Location Status Strip */}
      <div className="mt-3 py-2 px-3 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center space-x-2">
          {meshData.has_mesh ? (
            <>
              <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
              <span className="text-slate-300 font-medium">Anatomical Region:</span>
              <span className="text-cyan-300 font-semibold bg-cyan-950/70 px-2 py-0.5 rounded border border-cyan-800/80">
                {meshData.anatomical_location || 'Segmented Lesion Compartment'}
              </span>
            </>
          ) : (
            <>
              <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
              <span className="text-emerald-300 font-medium">Parenchyma Clear:</span>
              <span className="text-slate-400">No active tumor detected in 3D reconstruction.</span>
            </>
          )}
        </div>

        {meshData.has_mesh && meshData.centroid_3d && (
          <div className="flex items-center space-x-3 text-[11px] font-mono text-slate-400">
            <span>
              Centroid: <strong className="text-white">X:{meshData.centroid_3d[0]}</strong> Y:<strong className="text-white">{meshData.centroid_3d[1]}</strong> Z:<strong className="text-white">{meshData.centroid_3d[2]}</strong>
            </span>
            <button
              onClick={handleFocusTumour}
              className="px-2 py-0.5 rounded bg-rose-950/70 hover:bg-rose-900 text-rose-300 border border-rose-800 transition-colors flex items-center space-x-1"
              title="Align camera directly onto lesion"
            >
              <Crosshair className="h-3 w-3" />
              <span>Focus Tumour</span>
            </button>
          </div>
        )}
      </div>

      {/* 3D Viewport Area */}
      <div className="relative mt-3 w-full h-[400px] rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-inner">
        <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

        {/* Viewport Control Badges & Camera Buttons */}
        <div className="absolute top-3 left-3 flex flex-wrap gap-1.5">
          <button
            onClick={() => setPresetView('iso')}
            className="px-2 py-1 text-[11px] font-medium rounded bg-slate-900/90 text-slate-300 hover:text-white border border-slate-700 backdrop-blur-sm shadow"
          >
            Isometric
          </button>
          <button
            onClick={() => setPresetView('axial')}
            className="px-2 py-1 text-[11px] font-medium rounded bg-slate-900/90 text-slate-300 hover:text-white border border-slate-700 backdrop-blur-sm shadow"
            title="View from Superior (Top) directly matching axial scan"
          >
            Axial (Top)
          </button>
          <button
            onClick={() => setPresetView('sagittal')}
            className="px-2 py-1 text-[11px] font-medium rounded bg-slate-900/90 text-slate-300 hover:text-white border border-slate-700 backdrop-blur-sm shadow"
            title="Side profile view"
          >
            Sagittal (Side)
          </button>
          <button
            onClick={() => setPresetView('coronal')}
            className="px-2 py-1 text-[11px] font-medium rounded bg-slate-900/90 text-slate-300 hover:text-white border border-slate-700 backdrop-blur-sm shadow"
            title="Frontal view (Anterior to Posterior)"
          >
            Coronal (Front)
          </button>
        </div>

        {/* Anatomical Compass Key Top Center */}
        <div className="absolute top-3 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-full bg-slate-950/80 border border-slate-800 text-[10px] font-mono text-slate-400 backdrop-blur-sm hidden sm:flex items-center space-x-3">
          <span><strong className="text-cyan-400">A</strong>: Anterior</span>
          <span className="text-slate-700">|</span>
          <span><strong className="text-slate-300">P</strong>: Posterior</span>
          <span className="text-slate-700">|</span>
          <span><strong className="text-slate-300">L</strong>: Left</span>
          <span className="text-slate-700">|</span>
          <span><strong className="text-slate-300">R</strong>: Right</span>
        </div>

        {/* Toggles on Right */}
        <div className="absolute top-3 right-3 flex flex-col gap-1.5">
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`px-2.5 py-1 text-[11px] rounded border backdrop-blur-sm flex items-center space-x-1.5 transition-colors ${
              autoRotate ? 'bg-cyan-950/80 text-cyan-300 border-cyan-700' : 'bg-slate-900/80 text-slate-400 border-slate-800'
            }`}
          >
            <RotateCcw className={`h-3 w-3 ${autoRotate ? 'animate-spin' : ''}`} />
            <span>Rotate</span>
          </button>

          <button
            onClick={() => setWireframe(!wireframe)}
            className={`px-2.5 py-1 text-[11px] rounded border backdrop-blur-sm flex items-center space-x-1.5 transition-colors ${
              wireframe ? 'bg-cyan-950/80 text-cyan-300 border-cyan-700' : 'bg-slate-900/80 text-slate-400 border-slate-800'
            }`}
          >
            <Box className="h-3 w-3" />
            <span>Wireframe</span>
          </button>

          <button
            onClick={() => setShowBrainReference(!showBrainReference)}
            className={`px-2.5 py-1 text-[11px] rounded border backdrop-blur-sm flex items-center space-x-1.5 transition-colors ${
              showBrainReference ? 'bg-cyan-950/80 text-cyan-300 border-cyan-700' : 'bg-slate-900/80 text-slate-400 border-slate-800'
            }`}
          >
            <Eye className="h-3 w-3" />
            <span>Brain Cage</span>
          </button>

          {meshData.has_mesh && (
            <>
              <button
                onClick={() => setShowSlicePlane(!showSlicePlane)}
                className={`px-2.5 py-1 text-[11px] rounded border backdrop-blur-sm flex items-center space-x-1.5 transition-colors ${
                  showSlicePlane ? 'bg-cyan-950/80 text-cyan-300 border-cyan-700' : 'bg-slate-900/80 text-slate-400 border-slate-800'
                }`}
                title="Toggle axial cutting slice reference plane"
              >
                <Layers className="h-3 w-3" />
                <span>Slice Plane</span>
              </button>

              <button
                onClick={() => setShowBeacon(!showBeacon)}
                className={`px-2.5 py-1 text-[11px] rounded border backdrop-blur-sm flex items-center space-x-1.5 transition-colors ${
                  showBeacon ? 'bg-amber-950/80 text-amber-300 border-amber-700' : 'bg-slate-900/80 text-slate-400 border-slate-800'
                }`}
                title="Toggle lesion centroid beacon reticle"
              >
                <Crosshair className="h-3 w-3" />
                <span>Target Pin</span>
              </button>
            </>
          )}
        </div>

        {/* Mesh Statistics HUD */}
        <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-300 backdrop-blur-sm">
          <span>Vertices: {meshData.vertex_count?.toLocaleString() || 0}</span>
          <span className="mx-2 text-slate-600">|</span>
          <span>Faces: {meshData.face_count?.toLocaleString() || 0}</span>
          <span className="mx-2 text-slate-600">|</span>
          <span className={meshData.is_genuine_3d ? "text-emerald-400" : "text-cyan-400"}>
            {meshData.is_genuine_3d ? "Volumetric 3D" : "Co-Registered 2D-to-3D"}
          </span>
        </div>
      </div>

      {/* Limitation Notice */}
      <div className="mt-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-start space-x-2.5 text-xs text-slate-400">
        <AlertCircle className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <span className="font-semibold text-slate-300">Spatial Accuracy Note: </span>
          {meshData.limitation_note || "Rotate via left click drag, zoom via scroll wheel, preset buttons select clinical viewpoints."}
        </p>
      </div>
    </div>
  );
};
