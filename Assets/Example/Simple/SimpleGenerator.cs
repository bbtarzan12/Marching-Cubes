using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
public class SimpleGenerator : MonoBehaviour
{
    [SerializeField] Vector3 chunkScale = new Vector3(1f, 1f, 1f);
    [SerializeField] Vector3Int cellSize = new Vector3Int(32, 32, 32);
    [Range(0.001f, 1f)] [SerializeField] float frequency = 0.05f;
    [Range(1, 5)] [SerializeField] int octaves = 3;
    [Range(1, 10)] [SerializeField] float offsetSpeed;
    [SerializeField] Material material;
    [SerializeField] bool enableJob;
    [SerializeField] bool enableTriangleIndexing;

    Vector3Int gridSize;
    Vector3 worldOffset;

    Voxel[,,] voxels;

    // Mesh
    Mesh mesh;
    MeshFilter meshFilter;
    MeshRenderer meshRenderer;

    // For Avoid GC
    List<Vector3> vertices = new List<Vector3>();
    List<int> triangles = new List<int>();
    List<Color> colors = new List<Color>();

    [BurstCompile]
    struct VoxelNoiseJob : IJobParallelFor
    {
        [ReadOnly] public Vector3Int cellSize;
        [ReadOnly] public Vector3Int gridSize;
        [ReadOnly] public Vector3 worldOffset;
        [ReadOnly] public float frequency;
        [ReadOnly] public int octaves;
        
        [WriteOnly] public NativeArray<Voxel> voxels;
        
        public void Execute(int index)
        {
            Vector3Int gridPosition = To3DIndex(index);
            if (gridPosition.x >= cellSize.x || gridPosition.y >= cellSize.y || gridPosition.z >= cellSize.z || gridPosition.x < 1 || gridPosition.y < 1 || gridPosition.z < 1)
                return;

            Vector3 worldPosition = gridPosition + worldOffset;

            Vector3 World2DPosition = new Vector3(worldPosition.x, worldPosition.z);
            
            float density = -worldPosition.y + 30f;
            density += Noise.Perlin2DFractal(World2DPosition, frequency, octaves) * 10f;
            density += Noise.Perlin2DFractal(World2DPosition, frequency * 0.5f, octaves) * 80f;
            float height = Mathf.Clamp01((worldPosition.y - 5f) / 30f);
            voxels[index] = new Voxel {Density = density, Height = height};
        }
        
        Vector3Int To3DIndex(int index)
        {
            return new Vector3Int 
            {			
                z = index % gridSize.z,
                y = (index / gridSize.z) % gridSize.y,
                x = index / (gridSize.y * gridSize.z)
            };
        }
    }
    
    void Awake()
    {
        gridSize = cellSize + Vector3Int.one;
        voxels = new Voxel[gridSize.x, gridSize.y, gridSize.z];
        
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();
        mesh = new Mesh {indexFormat = IndexFormat.UInt32};
    }

    void Start()
    {
        meshFilter.mesh = mesh;
        meshRenderer.material = material;
    }

    void Update()
    {
        worldOffset += Time.deltaTime * offsetSpeed * new Vector3(1, 0, 1);
        GenerateVoxelDensity();
        GenerateMesh();
    }

    void GenerateVoxelDensity()
    {
        if (enableJob)
        {
            NativeArray<Voxel> nativeVoxels = new NativeArray<Voxel>(gridSize.x * gridSize.y * gridSize.z, Allocator.TempJob);

            VoxelNoiseJob noiseJob = new VoxelNoiseJob
            {
                cellSize = cellSize,
                gridSize = gridSize,
                voxels = nativeVoxels,
                frequency = frequency,
                octaves = octaves,
                worldOffset = worldOffset
            };
            JobHandle noiseJobHandle = noiseJob.Schedule(gridSize.x * gridSize.y * gridSize.z, 32);
            noiseJobHandle.Complete();
            
            unsafe
            {
                fixed (void* voxelPointer = voxels)
                {
                    UnsafeUtility.MemCpy(voxelPointer, NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(nativeVoxels), voxels.Length * (long) UnsafeUtility.SizeOf<Voxel>());
                }
            }

            nativeVoxels.Dispose();
        }
        else
        {
            for (int x = 1; x < cellSize.x; x++)
            {
                for (int y = 1; y < cellSize.y; y++)
                {
                    for (int z = 1; z < cellSize.z; z++)
                    {
                        Vector3 worldPosition = new Vector3(x, y, z) + worldOffset;

                        Vector3 World2DPosition = new Vector3(worldPosition.x, worldPosition.z);
            
                        float density = -worldPosition.y + 30f;
                        density += Noise.Perlin2DFractal(World2DPosition, frequency, octaves) * 10f;
                        density += Noise.Perlin2DFractal(World2DPosition, frequency * 0.5f, octaves) * 80f;
                        float height = Mathf.Clamp01((worldPosition.y - 5f) / 30f);

                        voxels[x, y, z].Density = density;
                        voxels[x, y, z].Height = height;
                    }
                }
            }   
        }
    }

    void GenerateMesh()
    {
        if (enableJob)
        {
            MarchingCubes.GenerateMarchingCubesWithJob(voxels, cellSize, chunkScale, enableTriangleIndexing, vertices, triangles, colors);
        }
        else
        {
            MarchingCubes.GenerateMarchingCubes(voxels, cellSize, chunkScale, enableTriangleIndexing, vertices, triangles, colors);
        }

        mesh.Clear();
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.SetColors(colors);
        mesh.RecalculateNormals();
    }

}
