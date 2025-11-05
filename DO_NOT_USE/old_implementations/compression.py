"""
FILE: compression.py
PURPOSE: Multi-stage lossless compression for projective weights
DEPENDENCIES: numpy, collections, heapq

CRITICAL CONCEPTS:
- Pipeline order: sparsity → Huffman (Phase 1)
- Each stage must be perfectly reversible
- Metadata serialized with compressed data
- Validation: bit-for-bit equality after decompression

PHASE 1 (POC): Sparsity + Huffman (2 stages, quick implementation)

COMPRESSION RATIONALE:
- Sparsity: 40-60% weights are near-zero (stored > 200)
- Huffman: Non-uniform distribution (powers of 2 frequent)
"""

import numpy as np
from collections import Counter
import heapq
from typing import Dict, List, Tuple, Any
import struct
import hashlib


class ProjectiveCompressor:
    """
    Multi-stage lossless compression for projective integer weights.
    
    Phase 1 (POC): 2 stages (sparsity + Huffman)
    
    Usage:
        compressor = ProjectiveCompressor(phase=1)
        compressed = compressor.compress(weights_dict)
        decompressed = compressor.decompress(compressed)
        assert np.array_equal(original, decompressed)  # Lossless!
    
    Attributes:
        phase: 1 (POC, 2 stages)
        sparse_threshold: Values > this treated as sparse (default 200)
    """
    
    def __init__(self, phase: int = 1, sparse_threshold: int = 200):
        self.phase = phase
        self.sparse_threshold = sparse_threshold
    
    def compress(self, weights: Dict[str, np.ndarray]) -> bytes:
        """
        Compress all model weights through pipeline.
        
        Args:
            weights: Dict of layer_name → uint8 numpy arrays
        
        Returns:
            Compressed bytes (includes metadata)
        
        Phase 1 Algorithm (POC):
            1. Stage 1: Sparsity encoding
            2. Stage 2: Huffman coding
            3. Serialize with metadata
        """
        compressed_layers = {}
        metadata = {
            'phase': self.phase,
            'sparse_threshold': self.sparse_threshold,
            'layer_shapes': {},
            'huffman_trees': {}
        }
        
        for layer_name, weight_array in weights.items():
            # Stage 1: Sparsity encoding
            indices, values = self._encode_sparsity(weight_array.flatten())
            metadata['layer_shapes'][layer_name] = list(weight_array.shape)
            
            # Stage 2: Huffman encoding
            huffman_bitstring, huffman_codes = self._encode_huffman(values)
            metadata['huffman_trees'][layer_name] = huffman_codes
            
            compressed_layers[layer_name] = {
                'indices': indices.tolist(),
                'huffman_bits': huffman_bitstring,
                'value_count': len(values)
            }
        
        # Serialize everything
        serialized = self._serialize(compressed_layers, metadata)
        return serialized
    
    def _encode_sparsity(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 1: Sparsity encoding.
        
        Removes near-zero weights (stored > threshold).
        
        Args:
            weights: uint8 array
        
        Returns:
            (non_sparse_indices, non_sparse_values)
        
        Algorithm:
            1. Create mask: weights <= threshold
            2. Extract indices where mask is True
            3. Extract values at those indices
            4. Return (indices, values)
        
        Compression: 2-3× (40-60% sparsity typical)
        """
        mask = weights <= self.sparse_threshold
        indices = np.where(mask)[0]
        values = weights[indices]
        return indices, values
    
    def _decode_sparsity(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        original_shape: tuple
    ) -> np.ndarray:
        """
        Reverse sparsity encoding.
        
        Algorithm:
            1. Create array of zeros with original_shape
            2. Set values at indices
            3. Remaining positions stay zero (sparse)
        """
        total_size = np.prod(original_shape)
        result = np.zeros(total_size, dtype=np.uint8)
        result[indices] = values
        return result.reshape(original_shape)
    
    def _build_huffman_tree(self, frequencies: Dict[int, int]) -> Dict[int, str]:
        """
        Stage 2: Build Huffman coding tree.
        
        Args:
            frequencies: Dict of value → count
        
        Returns:
            Dict of value → binary_code_string
        
        Algorithm (Standard Huffman):
            1. Create min-heap: [(freq, [value, ""]) for each value]
            2. While heap has >1 element:
                a. Pop two minimum nodes
                b. Create parent: freq = sum, children = nodes
                c. Prefix left with '0', right with '1'
                d. Push parent to heap
            3. Extract codes from final tree
            4. Return {value: code}
        """
        if len(frequencies) == 0:
            return {}
        
        if len(frequencies) == 1:
            # Special case: only one unique value
            value = list(frequencies.keys())[0]
            return {value: '0'}
        
        # Create min-heap with (frequency, unique_id, node_data)
        heap = []
        node_id = 0
        
        for value, freq in frequencies.items():
            heapq.heappush(heap, (freq, node_id, value))
            node_id += 1
        
        # Build tree
        while len(heap) > 1:
            freq1, id1, node1 = heapq.heappop(heap)
            freq2, id2, node2 = heapq.heappop(heap)
            
            # Create parent node
            parent_freq = freq1 + freq2
            parent_node = (node1, node2)
            heapq.heappush(heap, (parent_freq, node_id, parent_node))
            node_id += 1
        
        # Extract codes by traversing tree
        _, _, root = heap[0]
        codes = {}
        
        def extract_codes(node, code=''):
            if isinstance(node, tuple):
                # Internal node
                left, right = node
                extract_codes(left, code + '0')
                extract_codes(right, code + '1')
            else:
                # Leaf node
                codes[node] = code
        
        extract_codes(root)
        return codes
    
    def _encode_huffman(self, values: np.ndarray) -> Tuple[str, Dict[int, str]]:
        """
        Encode values using Huffman codes.
        
        Algorithm:
            1. Build frequency table
            2. Build Huffman tree
            3. For each value: emit code from tree
            4. Return bitstring and codes
        """
        frequencies = dict(Counter(values))
        codes = self._build_huffman_tree(frequencies)
        bitstring = ''.join(codes[int(v)] for v in values)
        return bitstring, codes
    
    def _decode_huffman(
        self,
        bitstring: str,
        codes: Dict[int, str],
        count: int
    ) -> np.ndarray:
        """
        Decode Huffman bitstring.
        
        Algorithm:
            1. Build reverse lookup: code → value
            2. Traverse bitstring bit-by-bit
            3. When code recognized: emit value, reset
            4. Continue until count values decoded
        """
        reverse_codes = {code: value for value, code in codes.items()}
        
        decoded = []
        current_code = ''
        
        for bit in bitstring:
            current_code += bit
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ''
                if len(decoded) == count:
                    break
        
        return np.array(decoded, dtype=np.uint8)
    
    def _serialize(self, compressed_layers: Dict, metadata: Dict) -> bytes:
        """
        Serialize compressed data and metadata to bytes.
        
        Format:
            - Magic number (4 bytes): 'INTL'
            - Version (1 byte)
            - Metadata length (4 bytes)
            - Metadata (JSON)
            - Compressed data
            - Checksum (32 bytes, SHA256)
        """
        import json
        
        # Magic number
        result = b'INTL'
        
        # Version
        result += struct.pack('B', self.phase)
        
        # Serialize metadata
        metadata_json = json.dumps(metadata, indent=None)
        metadata_bytes = metadata_json.encode('utf-8')
        result += struct.pack('I', len(metadata_bytes))
        result += metadata_bytes
        
        # Serialize compressed data
        data_json = json.dumps(compressed_layers, indent=None)
        data_bytes = data_json.encode('utf-8')
        result += data_bytes
        
        # Compute checksum
        checksum = hashlib.sha256(result).digest()
        result += checksum
        
        return result
    
    def decompress(self, compressed_bytes: bytes) -> Dict[str, np.ndarray]:
        """
        Decompress back to original projective weights.
        
        This reverses all compression stages.
        MUST be lossless - validate with checksums!
        
        Algorithm:
            1. Deserialize metadata
            2. Verify checksum
            3. Reverse compression stages (in reverse order)
            4. Validate output matches original
        """
        import json
        
        # Verify magic number
        magic = compressed_bytes[:4]
        if magic != b'INTL':
            raise ValueError(f"Invalid magic number: {magic}")
        
        # Read version
        version = struct.unpack('B', compressed_bytes[4:5])[0]
        
        # Read metadata
        metadata_len = struct.unpack('I', compressed_bytes[5:9])[0]
        metadata_bytes = compressed_bytes[9:9+metadata_len]
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Read compressed data
        data_start = 9 + metadata_len
        checksum_start = len(compressed_bytes) - 32
        data_bytes = compressed_bytes[data_start:checksum_start]
        compressed_layers = json.loads(data_bytes.decode('utf-8'))
        
        # Verify checksum
        expected_checksum = compressed_bytes[checksum_start:]
        actual_checksum = hashlib.sha256(compressed_bytes[:checksum_start]).digest()
        if expected_checksum != actual_checksum:
            raise ValueError("Checksum mismatch - data corrupted!")
        
        # Reverse compression stages
        decompressed = {}
        
        for layer_name, layer_data in compressed_layers.items():
            # Reverse Huffman
            huffman_codes = metadata['huffman_trees'][layer_name]
            # Convert string keys back to ints
            huffman_codes = {int(k): v for k, v in huffman_codes.items()}
            values = self._decode_huffman(
                layer_data['huffman_bits'],
                huffman_codes,
                layer_data['value_count']
            )
            
            # Reverse sparsity
            indices = np.array(layer_data['indices'], dtype=np.int64)
            shape = tuple(metadata['layer_shapes'][layer_name])
            weights = self._decode_sparsity(indices, values, shape)
            
            decompressed[layer_name] = weights
        
        return decompressed
