using System.Collections.Generic;
using UnityEngine;

public class TerrainGenerator : MonoBehaviour
{
    Dictionary<Vector3Int, TerrainChunk> chunks = new Dictionary<Vector3Int, TerrainChunk>();
    [SerializeField] Vector3Int cellSize = new Vector3Int(32, 32, 32);
    [Range(0.001f, 1f)] [SerializeField] float frequency = 0.05f;
    [SerializeField] Vector3 chunkScale = Vector3.one; 
    [SerializeField] Vector3Int chunkSpawnSize = Vector3Int.one * 3;
    [SerializeField] int maxGenerateChunksInFrame = 5;
    [SerializeField] int maxUpdateChunksInFrame = 3;

    [SerializeField] bool enableDebug = false;
    [SerializeField] bool enableJob = true;
    [SerializeField] bool enableTriangleIndexing = true;
    
    [SerializeField] Material terrainMaterial;
    [SerializeField] Transform target;

    Vector3Int lastTargetChunkPosition = new Vector3Int(int.MinValue, int.MaxValue, int.MinValue);
    HashSet<Vector3Int> generateChunkSet = new HashSet<Vector3Int>();
    Queue<Vector3Int> generateChunkQueue = new Queue<Vector3Int>();
    Queue<TerrainChunk> updateChunkQueue = new Queue<TerrainChunk>();

    public Vector3Int CellSize => cellSize;
    public Vector3 ChunkScale => chunkScale;
    public bool EnableJob => enableJob;
    public bool EnableTriangleIndexing => enableTriangleIndexing;
    public Material TerrainMaterial => terrainMaterial;
    public float Frequency => frequency;
    public bool EnableDebug => enableDebug;


    void Update()
    {
        GenerateChunkByTargetPosition();
        UpdateChunkMesh();
        
        ProcessGenerateChunkQueue();
        ProcessUpdateChunkQueue();
    }

    void GenerateChunkByTargetPosition()
    {
        if (target == null)
            return;
        
        Vector3Int chunkPosition = WorldToChunk(target.position);

        if (lastTargetChunkPosition == chunkPosition)
            return;

        for (int x = chunkPosition.x - chunkSpawnSize.x; x <= chunkPosition.x + chunkSpawnSize.x; x++)
        {
            for (int y = chunkPosition.y - chunkSpawnSize.y; y <= chunkPosition.y + chunkSpawnSize.y; y++)
            {
                for (int z = chunkPosition.z - chunkSpawnSize.z; z <= chunkPosition.z + chunkSpawnSize.z; z++)
                {
                    Vector3Int newChunkPosition = new Vector3Int(x, y, z);
                    if (chunks.ContainsKey(newChunkPosition))
                        continue;
                    
                    if(generateChunkSet.Contains(newChunkPosition))
                        continue;
                    
                    generateChunkQueue.Enqueue(newChunkPosition);
                    generateChunkSet.Add(newChunkPosition);
                }
            }
        }

        lastTargetChunkPosition = chunkPosition;
    }
    
    void UpdateChunkMesh()
    {
        foreach (TerrainChunk chunk in chunks.Values)
        {
            if (!chunk.Dirty)
                continue;

            updateChunkQueue.Enqueue(chunk);
            
        }
    }

    void ProcessGenerateChunkQueue()
    {
        int numChunks = 0;
        while (generateChunkQueue.Count != 0)
        {
            if (numChunks >= maxGenerateChunksInFrame)
                return;

            Vector3Int chunkPosition = generateChunkQueue.Dequeue();
            GenerateChunk(chunkPosition);
            generateChunkSet.Remove(chunkPosition);
            numChunks++;
        }
    }
    
    void ProcessUpdateChunkQueue()
    {
        int numChunks = 0;
        while (updateChunkQueue.Count != 0)
        {
            if (numChunks >= maxUpdateChunksInFrame)
                return;
            
            TerrainChunk chunk = updateChunkQueue.Dequeue();
            chunk.UpdateMesh();
        }
    }
    
    TerrainChunk GenerateChunk(Vector3Int chunkPosition)
    {
        if (chunks.ContainsKey(chunkPosition))
            return chunks[chunkPosition];

        GameObject chunkGameObject = new GameObject(chunkPosition.ToString());
        chunkGameObject.transform.SetParent(transform);
        chunkGameObject.transform.position = ChunkToWorld(chunkPosition);

        TerrainChunk newChunk = chunkGameObject.AddComponent<TerrainChunk>();
        newChunk.Init(chunkPosition, this);

        chunks.Add(chunkPosition, newChunk);
        return newChunk;
    }

    TerrainChunk GetChunk(Vector3Int chunkPosition)
    {
	    if (chunks.ContainsKey(chunkPosition))
		    return chunks[chunkPosition];

	    return null;
    }

    public void AddVoxelSphere(Vector3 worldPosition, float density, float radius)
    {
        for (float x = -radius; x < radius; x++)
        {
            for (float y = -radius; y < radius; y++)
            {
                for (float z = -radius; z < radius; z++)
                {
                    Vector3 offset = new Vector3(x, y, z);
                    float distance = offset.magnitude;

                    if (distance > radius)
                        continue;
                    
                    Vector3 spherePosition = new Vector3(x, y, z) + worldPosition;
                    AddVoxel(spherePosition, density);
                }
            }
        }
    }

    public void AddVoxel(Vector3 worldPosition, float density)
    {
        AddVoxel(WorldToGrid(worldPosition), density);
    }
    
    void AddVoxel(Vector3Int gridPosition, float density)
    {
	    Vector3Int chunkPosition = GridtoChunk(gridPosition);

	    TerrainChunk terrainChunk = GetChunk(chunkPosition);

	    if (terrainChunk == null)
	    {
		    GenerateChunk(chunkPosition);
		    return;
	    }

	    terrainChunk.AddVoxel(gridPosition, density);

	    TerrainChunk neighborChunk;
	    if (Mod(gridPosition.x, cellSize.x) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x - 1, chunkPosition.y, chunkPosition.z));
		    neighborChunk.AddVoxel(gridPosition, density);
            
		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x + 1, chunkPosition.y, chunkPosition.z));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }

	    if (Mod(gridPosition.y, cellSize.y) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x, chunkPosition.y - 1, chunkPosition.z));
		    neighborChunk.AddVoxel(gridPosition, density);

		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x, chunkPosition.y + 1, chunkPosition.z));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }

	    if (Mod(gridPosition.z, cellSize.z) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x, chunkPosition.y, chunkPosition.z + 1));
		    neighborChunk.AddVoxel(gridPosition, density);
            
		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x, chunkPosition.y, chunkPosition.z - 1));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }

	    if (Mod(gridPosition.x, cellSize.x) == 0 && Mod(gridPosition.y, cellSize.y) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x + 1, chunkPosition.y + 1, chunkPosition.z));
		    neighborChunk.AddVoxel(gridPosition, density);
            
		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x - 1, chunkPosition.y - 1, chunkPosition.z));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }

	    if (Mod(gridPosition.x, cellSize.x) == 0 && Mod(gridPosition.z, cellSize.z) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x + 1, chunkPosition.y, chunkPosition.z + 1));
		    neighborChunk.AddVoxel(gridPosition, density);
            
		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x - 1, chunkPosition.y, chunkPosition.z - 1));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }

	    if (Mod(gridPosition.y, cellSize.y) == 0 && Mod(gridPosition.z, cellSize.z) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x, chunkPosition.y + 1, chunkPosition.z + 1));
		    neighborChunk.AddVoxel(gridPosition, density);
            
		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x, chunkPosition.y - 1, chunkPosition.z - 1));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }

	    if (Mod(gridPosition.x, cellSize.x) == 0 && Mod(gridPosition.y, cellSize.y) == 0 && Mod(gridPosition.z, cellSize.z) == 0)
	    {
            neighborChunk = GetChunk(new Vector3Int(chunkPosition.x + 1, chunkPosition.y + 1, chunkPosition.z + 1));
		    neighborChunk.AddVoxel(gridPosition, density);
            
		    neighborChunk = GetChunk(new Vector3Int(chunkPosition.x - 1, chunkPosition.y - 1, chunkPosition.z - 1));
		    neighborChunk.AddVoxel(gridPosition, density);

	    }
    }


    public Vector3 ChunkToWorld(Vector3Int chunkPosition)
    {
        return Vector3.Scale(ChunkToWorldNotScaled(chunkPosition), chunkScale);
    }

    public Vector3 ChunkToWorldCenter(Vector3Int chunkPosition)
    {
        return Vector3.Scale(Vector3.Scale(Vector3.one * 0.5f + chunkPosition, cellSize), chunkScale);
    }

    public Vector3 ChunkToWorldNotScaled(Vector3Int chunkPosition)
    {
        return Vector3.Scale(chunkPosition, cellSize);
    }

    public Vector3 GridToWorld(Vector3Int gridPosition)
    {
        return Vector3.Scale(gridPosition, chunkScale);
    }

    public Vector3Int WorldToGrid(Vector3 worldPosition)
    {
        return new Vector3Int
        {
            x = Mathf.RoundToInt(worldPosition.x  / chunkScale.x),
            y = Mathf.RoundToInt(worldPosition.y / chunkScale.y),
            z = Mathf.RoundToInt(worldPosition.z / chunkScale.z)
        };
    }

    public Vector3Int WorldToChunk(Vector3 worldPosition)
    {
        return new Vector3Int
        {
            x = Mathf.FloorToInt(worldPosition.x / cellSize.x / chunkScale.x),
            y = Mathf.FloorToInt(worldPosition.y / cellSize.y / chunkScale.y),
            z = Mathf.FloorToInt(worldPosition.z / cellSize.z / chunkScale.z)
        };
    }

    public Vector3Int GridtoChunk(Vector3Int gridPosition)
    {
        return new Vector3Int
        {
            x = Mathf.FloorToInt((float) gridPosition.x / cellSize.x),
            y = Mathf.FloorToInt((float) gridPosition.y / cellSize.y),
            z = Mathf.FloorToInt((float) gridPosition.z / cellSize.z)
        };
    }
    
    int Mod(int x, int m) 
    {
	    int r = x%m;
	    return r<0 ? r+m : r;
    }

    void OnDrawGizmos()
    {
        if (!enableDebug)
            return;
        
        Vector3Int[] enqueuedChunks = generateChunkQueue.ToArray();
        foreach (Vector3Int chunkPosition in enqueuedChunks)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(ChunkToWorldCenter(chunkPosition), Vector3.Scale(CellSize, ChunkScale));
        }
    }
}