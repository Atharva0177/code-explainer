import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import time
import os
import sys
import platform
from pathlib import Path
import requests
import ast
import tokenize
from io import StringIO
import re
from collections import Counter
import pandas as pd
import base64
# Add these imports at the top of the file
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image
# Optional dependencies with graceful fallbacks
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    st.warning("Warning: gTTS not installed. Text-to-speech disabled.")
    st.info("Install with: `pip install gtts`")
    HAS_GTTS = False

class ESP32Manager:
    """Class to handle communication with ESP32 OLED display"""
    def __init__(self):
        self.esp32_ip = None
        self.esp32_port = 80
        self.connected = False

    def setup_connection(self, ip=None):
        """Configure connection to ESP32"""
        if ip:
            self.esp32_ip = ip
        else:
            # If IP not provided, ask user
            self.esp32_ip = st.text_input("Enter ESP32 IP address:")
            if not self.esp32_ip:
                return False

        # Test connection
        try:
            with st.spinner("Testing connection to ESP32..."):
                # Simple ping to check connectivity
                requests.get(f"http://{self.esp32_ip}/ping", timeout=2)
                self.connected = True
                st.success(f"✓ Connected to ESP32 at {self.esp32_ip}")
                return True
        except Exception as e:
            st.error(f"✗ Failed to connect to ESP32: {e}")
            self.connected = False
            return False

    def send_explanation(self, explanation_text):
        """Send explanation text to ESP32 for OLED display"""
        if not self.connected:
            st.warning("Not connected to ESP32")
            return False

        try:
            # Split text into chunks that will fit on OLED display
            chunks = self._chunk_text(explanation_text)
            
            # Display chunk information for debugging
            st.info(f"Prepared {len(chunks)} text chunks for ESP32 OLED display")
            with st.expander("View text chunks"):
                for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                    st.code(f"Chunk {i+1}: {chunk}")
                if len(chunks) > 5:
                    st.text(f"...and {len(chunks)-5} more chunks")
            
            with st.spinner(f"Sending explanation to ESP32 ({len(chunks)} chunks)..."):
                # Send a simpler setup message first to ensure communication
                setup_data = {
                    "total_chunks": len(chunks),
                    "message": "Starting text transfer"
                }

                setup_url = f"http://{self.esp32_ip}/setup"
                try:
                    response = requests.post(setup_url, json=setup_data, timeout=5)
                    st.write(f"Setup response: {response.status_code} - {response.text}")

                    if response.status_code != 200:
                        st.error(f"ESP32 setup error: {response.text}")
                        return False
                except requests.RequestException as e:
                    st.error(f"Setup request failed: {e}")
                    return False

                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Track successful and failed chunks
                successful_chunks = 0
                failed_chunks = 0
                
                # Send each chunk with a short delay
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "chunk_num": i+1,
                        "text": chunk,
                        "total": len(chunks)  # Add total for ESP32 reference
                    }

                    chunk_url = f"http://{self.esp32_ip}/text"
                    try:
                        response = requests.post(chunk_url, json=chunk_data, timeout=5)
                        
                        if response.status_code == 200:
                            successful_chunks += 1
                        else:
                            st.error(f"Error sending chunk {i+1}: {response.text}")
                            failed_chunks += 1
                            
                    except requests.RequestException as e:
                        st.error(f"Request error on chunk {i+1}: {e}")
                        failed_chunks += 1

                    # Update progress bar
                    progress_bar.progress((i + 1) / len(chunks))
                    time.sleep(0.2)  # Reduced delay to prevent timeouts but still give ESP32 time

                if failed_chunks > 0:
                    st.warning(f"Completed with {successful_chunks} successful chunks and {failed_chunks} failed chunks")
                else:
                    st.success(f"Successfully sent all {len(chunks)} chunks to ESP32 OLED")
                
                # Send a final "complete" message
                try:
                    complete_data = {"status": "complete"}
                    requests.post(f"http://{self.esp32_ip}/complete", json=complete_data, timeout=5)
                except:
                    pass  # Ignore errors on completion message
                    
                return failed_chunks == 0

        except Exception as e:
            st.error(f"Error sending to ESP32: {e}")
            return False
        



    def test_display(self):
        """Send a simple test message to verify display functionality"""
        if not self.connected:
            st.warning("Not connected to ESP32")
            return False
            
        try:
            test_message = "ESP32 OLED\nTest Message\nConnection OK\n" + time.strftime("%H:%M:%S")
            
            # Just send a single chunk for testing
            test_data = {
                "chunk_num": 1,
                "text": test_message,
                "total": 1,
                "is_test": True
            }
            
            with st.spinner("Sending test message to ESP32..."):
                response = requests.post(f"http://{self.esp32_ip}/text", json=test_data, timeout=5)
                
                if response.status_code == 200:
                    st.success("Test message sent successfully!")
                    return True
                else:
                    st.error(f"Failed to send test message: {response.text}")
                    return False
        except Exception as e:
            st.error(f"Test message error: {e}")
            return False

    def _chunk_text(self, text, max_lines=4, max_chars_per_line=21):
        """Split text into chunks suitable for OLED display"""
        chunks = []
        current_chunk = []

        # First split by sentences to maintain coherence
        sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")

        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue

            # If sentence is too long, split into words
            if len(sentence) > max_chars_per_line:
                words = sentence.split()
                current_line = ""

                for word in words:
                    # If adding word exceeds line length, start new line
                    if len(current_line) + len(word) + 1 > max_chars_per_line:
                        current_chunk.append(current_line)
                        current_line = word

                        # If chunk is full, add to chunks and start new
                        if len(current_chunk) == max_lines:
                            chunks.append("\n".join(current_chunk))
                            current_chunk = []
                    else:
                        # Add word to current line
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word

                # Add remaining line if not empty
                if current_line:
                    current_chunk.append(current_line)

                    # If chunk is full, add to chunks and start new
                    if len(current_chunk) == max_lines:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
            else:
                # Sentence fits on one line
                current_chunk.append(sentence)

                # If chunk is full, add to chunks and start new
                if len(current_chunk) == max_lines:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []

        # Add final chunk if not empty
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

class CodeExplainer:
    def __init__(self, model_name="Atharva177/T5_new"):
        """Initialize the code explainer with specified model."""
        self.model_name = model_name
        self.device = None
        self.esp32 = ESP32Manager()
        
        # Initialize model
        self._setup_model()
        
    def _setup_model(self):
        """Load the HuggingFace model for code explanation."""
        with st.spinner("Loading model... (this may take a moment)"):
            try:
                # Add from_tf=True to load TensorFlow weights
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, from_tf=True)
                self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
                
                # Move model to GPU if available
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                st.success(f"✓ Model loaded successfully and running on {self.device}")
            except Exception as e:
                st.error(f"✗ Error loading model: {e}")
                st.error("Make sure you have an internet connection and the model name is correct.")
                raise e

    # Fix 3: Fix the create_audio_player method
    def create_audio_player(self, text):
        """Create audio player for spoken explanation."""
        if not HAS_GTTS:
            st.warning("Text-to-speech unavailable - gTTS not installed")
            return
            
        try:
            # Create audio file with gTTS
            with st.spinner("Generating speech..."):
                output_file = "explanation.mp3"
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(output_file)
                
                # Create audio player with direct file reading
                with open(output_file, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name="explanation.mp3",
                        mime="audio/mpeg"
                    )
        except Exception as e:
            st.error(f"✗ TTS Error: {e}")
            
    def _clean_text(self, text):
        """Basic text cleaning for model output."""
        # Basic cleanup
        text = text.strip()
        
        # Remove unnecessary whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Ensure proper spacing after periods
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Fix line breaks before numbered items
        text = re.sub(r'(\d+\.)([A-Z])', r'\1 \2', text)
        
        return text
        
    def _check_for_repetition(self, text):
        """Check for and remove repeated phrases and sentences."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Track sentences we've seen
        seen_sentences = set()
        unique_sentences = []
        
        for sentence in sentences:
            # Normalize for comparison (remove whitespace and lowercase)
            norm_sentence = re.sub(r'\s+', ' ', sentence.strip().lower())
            
            # Skip if we've seen this exact sentence or a very similar one before
            if norm_sentence and norm_sentence not in seen_sentences and len(norm_sentence) > 5:
                seen_sentences.add(norm_sentence)
                unique_sentences.append(sentence)
        
        # Rejoin the cleaned sentences
        cleaned_text = ' '.join(unique_sentences)
        
        # Fix any spacing issues
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s*([.,;:!?])', r'\1', cleaned_text)
        cleaned_text = re.sub(r'([.,;:!?])\s*', r'\1 ', cleaned_text).strip()
        
        return cleaned_text
        
    def _format_explanation(self, text):
        """Format the explanation to make it more readable."""
        # First clean the text
        text = self._clean_text(text)
        
        # Check for and remove repetitions
        text = self._check_for_repetition(text)
        
        # Structure the explanation
        lines = []
        current_line = ""
        
        # Split text into lines and paragraphs
        paragraphs = text.split('. ')
        for i, para in enumerate(paragraphs):
            # Skip empty paragraphs
            if not para.strip():
                continue
                
            # Add a period back except for the last paragraph if it ends with ! or ?
            if i < len(paragraphs) - 1 or not para.rstrip().endswith(('!', '?')):
                para = para + '.'
                
            # Start a new line for numbered items
            if re.match(r'^\d+\.', para):
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                lines.append(para)
            else:
                if current_line:
                    current_line += ' ' + para
                else:
                    current_line = para
                    
        # Add the last line if there is one
        if current_line:
            lines.append(current_line)
            
        # Join lines with newlines
        formatted_text = '\n\n'.join(lines)
        
        # Ensure proper line breaks for lists
        formatted_text = re.sub(r'(\d+\..*?\.)\s+(\d+\.)', r'\1\n\n\2', formatted_text)
        
        return formatted_text

    def explain_code(self, code_snippet):
        """Use model to generate explanation for the provided code."""
        # For very large code snippets, split into sections
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        if len(code_snippet.splitlines()) > 30:
            status_text.text("Code is large - generating section-by-section explanation...")
            
            # Split into logical chunks
            chunks = self._split_code_into_chunks(code_snippet)
            
            # Generate explanation for each chunk
            explanations = []
            for i, chunk in enumerate(chunks):
                progress_bar.progress((i / len(chunks)) * 0.9)  # Leave 10% for final formatting
                status_text.text(f"Explaining section {i+1}/{len(chunks)}...")
                
                # Generate explanation for this chunk
                chunk_explanation = self._generate_explanation(chunk)
                explanations.append(chunk_explanation)
            
            # Combine explanations with section markers
            status_text.text("Combining explanations...")
            combined = self._combine_chunked_explanations(chunks, explanations)
            progress_bar.progress(1.0)
            status_text.text("Explanation generation complete!")
            
            return combined
        else:
            # For smaller code snippets, generate in one go
            status_text.text("Generating explanation...")
            explanation = self._generate_explanation(code_snippet)
            progress_bar.progress(1.0)
            status_text.text("Explanation generation complete!")
            
            return explanation

    def _generate_explanation(self, code_snippet):
        """Internal method to generate explanation for a code snippet."""
        input_text = f"Explain this code thoroughly: {code_snippet}"
        
        # Tokenize with reasonable length for the model
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=4096,        # More conservative length
                    num_beams=10,
                    no_repeat_ngram_size=3, # Prevent repeating phrases
                    early_stopping=False,    # Stop when decent explanation found
                    do_sample=True,        # Enable sampling for more natural language
                    temperature=0.7,        # Slightly randomized for natural language
                    repetition_penalty=1.2  # Discourage repetition
                )
            
            # Decode explanation
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Format the explanation
            explanation = self._format_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            st.warning(f"Error during explanation generation: {str(e)}")
            # Fallback to simpler parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,
                    temperature=0.7
                )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            explanation = self._format_explanation(explanation)
            
            return explanation

    def _split_code_into_chunks(self, code):
        """Split code into logical chunks for processing."""
        try:
            # Try to parse the code to get a proper AST
            parsed = ast.parse(code)
            
            # Get top-level code blocks
            chunks = []
            current_chunk = []
            current_chunk_lines = 0
            
            for node in parsed.body:
                node_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
                
                # If this node would make the chunk too big, start a new chunk
                if current_chunk_lines + node_lines > 25:  # Aim for ~25 lines per chunk
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_chunk_lines = 0
                
                # Add the code for this node
                node_code = ast.get_source_segment(code, node)
                if node_code:
                    current_chunk.append(node_code)
                    current_chunk_lines += node_lines
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            
            return chunks if chunks else [code]  # Return the original if no chunks
            
        except SyntaxError:
            # If parsing fails, fall back to simpler line-based chunking
            lines = code.splitlines()
            chunks = []
            
            # Try to split on logical boundaries like blank lines
            current_chunk = []
            for line in lines:
                current_chunk.append(line)
                
                # If we have enough lines or hit a blank line after some content
                if (len(current_chunk) >= 25 or 
                    (not line.strip() and len(current_chunk) > 5)):
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
            
            # Add the last chunk
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                
            return chunks if chunks else [code]  # Return the original if no chunks

    def _combine_chunked_explanations(self, code_chunks, explanations):
        """Combine explanations from multiple chunks into a coherent whole."""
        if len(explanations) == 1:
            return explanations[0]
            
        # Create a combined explanation with section headers
        combined = "# Complete Code Explanation\n\n"
        
        for i, (chunk, explanation) in enumerate(zip(code_chunks, explanations)):
            # Get a descriptive name for this chunk
            chunk_name = self._get_chunk_description(chunk, i+1)
            
            # Add a section header
            combined += f"## Section {i+1}: {chunk_name}\n\n"
            combined += explanation.strip() + "\n\n"
        
        # Add an overall summary
        combined += "\n## Overall Summary\n\n"
        combined += "The code above performs the following functions:\n"
        
        # Extract key points from each explanation to create summary bullets
        for i, explanation in enumerate(explanations):
            key_point = self._extract_key_point(explanation)
            combined += f"- Section {i+1}: {key_point}\n"
        
        return combined

    def _get_chunk_description(self, chunk, index):
        """Generate a descriptive name for a code chunk."""
        # Try to identify what this chunk contains
        if "class " in chunk:
            class_match = re.search(r'class\s+([A-Za-z0-9_]+)', chunk)
            if class_match:
                return f"Class '{class_match.group(1)}'"
        
        if "def " in chunk:
            func_match = re.search(r'def\s+([A-Za-z0-9_]+)', chunk)
            if func_match:
                return f"Function '{func_match.group(1)}'"
        
        # If no class or function, try to describe what it does
        first_line = chunk.splitlines()[0].strip()
        if first_line.startswith('#'):
            # Use the comment as description
            return first_line.lstrip('#').strip()
        
        # Default descriptor
        return f"Code Block {index}"

    def _extract_key_point(self, explanation):
        """Extract a key point from an explanation for the summary."""
        # Try to get the first sentence that's not too short
        sentences = re.split(r'(?<=[.!?])\s+', explanation)
        
        for sentence in sentences:
            if len(sentence.split()) >= 5:
                # Clean up the sentence
                cleaned = sentence.strip()
                if cleaned.startswith("This code"):
                    return cleaned
                if cleaned.startswith("The code"):
                    return cleaned
                return cleaned
        
        # Fallback
        return "Contains code functionality"

    def analyze_code_complexity(self, code):
        """Analyze the complexity of the provided code."""
        try:
            st.write("### Code Complexity Analysis")
            
            # Basic metrics
            lines = code.splitlines()
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
            metrics = {
                "Lines of Code": len(lines),
                "Lines of Code (excluding comments & blank lines)": len(code_lines),
            }
            
            # Try to parse the code to analyze structure
            try:
                parsed = ast.parse(code)
                
                # Count functions and classes
                functions = [node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(parsed) if isinstance(node, ast.ClassDef)]
                
                metrics["Functions"] = len(functions)
                metrics["Classes"] = len(classes)
                
                # Count control structures
                if_statements = len([node for node in ast.walk(parsed) if isinstance(node, ast.If)])
                for_loops = len([node for node in ast.walk(parsed) if isinstance(node, ast.For)])
                while_loops = len([node for node in ast.walk(parsed) if isinstance(node, ast.While)])
                try_except = len([node for node in ast.walk(parsed) if isinstance(node, ast.Try)])
                
                metrics["Conditional Statements"] = if_statements
                metrics["Loops"] = for_loops + while_loops
                metrics["Try/Except Blocks"] = try_except
                
                # Count variable assignments
                assignments = len([node for node in ast.walk(parsed) if isinstance(node, ast.Assign)])
                metrics["Variable Assignments"] = assignments
                
                # Maximum nesting depth
                max_depth = self._get_max_nesting_depth(parsed)
                metrics["Maximum Nesting Depth"] = max_depth
                
                # Calculate cyclomatic complexity
                cyclomatic_complexity = (
                    1 +  # Base complexity
                    if_statements +  # if statements
                    len([node for node in ast.walk(parsed) if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)]) +  # and operators
                    len([node for node in ast.walk(parsed) if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)]) +  # or operators
                    for_loops +  # for loops
                    while_loops +  # while loops
                    len([node for node in ast.walk(parsed) if isinstance(node, ast.ExceptHandler)])  # except blocks
                )
                metrics["Cyclomatic Complexity"] = cyclomatic_complexity
                
                # Cognitive complexity (simplified estimate)
                cognitive_complexity = cyclomatic_complexity + max_depth * 2
                metrics["Cognitive Complexity (estimated)"] = cognitive_complexity
                
                # Complexity assessment
                if cyclomatic_complexity <= 5:
                    risk = "Low - Simple, well-structured code"
                elif cyclomatic_complexity <= 10:
                    risk = "Moderate - Reasonably complex"
                elif cyclomatic_complexity <= 20:
                    risk = "High - Complex code that may need refactoring"
                else:
                    risk = "Very High - Code should be refactored into smaller functions"
                    
                metrics["Complexity Assessment"] = risk
                
            except SyntaxError as e:
                st.warning(f"Note: Could not parse code structure due to syntax error: {e}")
            
            # Display metrics in a table
            metrics_df = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Value": list(metrics.values())
            })
            # Ensure all values are treated as objects (strings) to prevent conversion issues
            metrics_df["Value"] = metrics_df["Value"].astype(str)
            st.table(metrics_df)
            
            # Complexity assessment
            if "Complexity Assessment" in metrics:
                risk_level = metrics["Complexity Assessment"]
                
                # Choose color based on risk level
                color = "green"
                if "Moderate" in risk_level:
                    color = "orange"
                elif "High" in risk_level or "Very High" in risk_level:
                    color = "red"
                    
                st.markdown(f"<h4 style='color: {color}'>Complexity Assessment:</h4>", unsafe_allow_html=True)
                st.info(risk_level)
            
            return metrics
            
        except Exception as e:
            st.error(f"Error analyzing code complexity: {e}")
            return {}

    def _get_max_nesting_depth(self, tree):
        """Calculate the maximum nesting depth in the AST."""
        max_depth = 0
        
        def measure_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            # Increment depth for nested control structures
            nested_structures = (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, nested_structures):
                    measure_depth(child, current_depth + 1)
                else:
                    measure_depth(child, current_depth)
        
        measure_depth(tree)
        return max_depth

    def highlight_important_code_elements(self, code, explanation):
        """Identify and highlight important elements in the code based on the explanation."""
        try:
            # Tokenize the code to extract identifiers and keywords
            code_elements = []
            tokens = list(tokenize.generate_tokens(StringIO(code).readline))
            
            for token in tokens:
                # Extract identifiers (variable names, function names, etc.)
                if token.type == tokenize.NAME and not token.string in ['True', 'False', 'None']:
                    code_elements.append(token.string)
            
            # Count how many times each element appears in the explanation
            explanation_words = explanation.lower().split()
            element_mentions = {}
            
            for element in set(code_elements):
                # Count mentions of this element in the explanation
                count = explanation_words.count(element.lower())
                if count > 0:
                    element_mentions[element] = count
            
            # Sort elements by mention frequency
            sorted_elements = sorted(element_mentions.items(), key=lambda x: x[1], reverse=True)
            
            # Get highlighted code
            highlighted_code = []
            significant_elements = [element for element, count in sorted_elements if count >= 2]
            
            if significant_elements:
                for line in code.splitlines():
                    highlighted = line
                    has_highlight = False
                    for element in significant_elements[:10]:  # Top 10 elements
                        # Simple regex to match whole words only
                        pattern = r'\b' + re.escape(element) + r'\b'
                        if re.search(pattern, highlighted):
                            has_highlight = True
                            # Use markdown to highlight
                            highlighted = re.sub(pattern, f"**`{element}`**", highlighted)
                    
                    # Add all lines, but mark highlighted ones
                    if has_highlight:
                        highlighted_code.append(f"→ {highlighted}")
                    else:
                        highlighted_code.append(f"  {highlighted}")
            else:
                # If no significant elements, just return the code
                highlighted_code = code.splitlines()
                
            return sorted_elements, "\n".join(highlighted_code)
        except Exception as e:
            st.error(f"Error analyzing code highlights: {e}")
            return [], ""
        




    def visualize_code_structure(self, code):
        """Generate interactive visualizations of code structure."""
        try:
            # Parse the code to get AST
            parsed = ast.parse(code)
            
            # Create tabs for different visualization types
            viz_tab1, viz_tab2 = st.tabs(["Function Flow", "Dependency Graph"])
            
            with viz_tab1:
                st.subheader("Code Flow Visualization")
                
                # Generate a flow chart using graphviz
                graph = graphviz.Digraph()
                graph.attr('node', shape='box', style='filled', fillcolor='lightblue')
                
                # Track nodes and edges
                nodes = set()
                edges = []
                
                # Find all function definitions and their calls
                functions = {}
                function_calls = {}
                
                # First pass: collect function definitions
                for node in ast.walk(parsed):
                    if isinstance(node, ast.FunctionDef):
                        functions[node.name] = node
                        nodes.add(node.name)
                        graph.node(node.name)
                
                # Second pass: collect function calls and build edges
                for node in ast.walk(parsed):
                    if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id in functions:
                        # Find the parent function containing this call
                        parent = None
                        for func_name, func_node in functions.items():
                            if func_node.lineno <= node.lineno <= func_node.end_lineno:
                                parent = func_name
                                break
                        
                        if parent and parent != node.func.id:  # Avoid self-references
                            edges.append((parent, node.func.id))
                
                # Add all edges to the graph
                for src, dest in edges:
                    graph.edge(src, dest)
                
                # Display the graph
                if nodes:
                    st.graphviz_chart(graph)
                else:
                    st.info("No function relationships detected in the code.")
                    
                # Add explanation
                st.caption("This diagram shows function calls between different parts of the code.")
                
            with viz_tab2:
                st.subheader("Module Dependency Graph")
                
                # Create a NetworkX graph for dependencies
                G = nx.DiGraph()
                
                # Track imports and module dependencies
                imports = []
                
                for node in ast.walk(parsed):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                            G.add_node(name.name, type="module")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                            G.add_node(node.module, type="module")
                            for name in node.names:
                                G.add_node(f"{node.module}.{name.name}", type="component")
                                G.add_edge(node.module, f"{node.module}.{name.name}")
                
                # Add the current file as the central node
                if imports:
                    file_name = "Current Code"
                    G.add_node(file_name, type="file")
                    
                    # Connect imports to the current file
                    for imp in imports:
                        G.add_edge(file_name, imp)
                    
                    # Calculate node sizes based on their connections
                    sizes = [300 * (len(list(G.neighbors(n))) + 1) for n in G.nodes()]
                    
                    # Generate the visualization
                    plt.figure(figsize=(10, 7))
                    pos = nx.spring_layout(G)
                    
                    # Color nodes by type
                    node_colors = []
                    for node in G.nodes():
                        if G.nodes[node].get('type') == 'file':
                            node_colors.append('lightgreen')
                        elif G.nodes[node].get('type') == 'module':
                            node_colors.append('skyblue')
                        else:
                            node_colors.append('lightgray')
                    
                    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                        node_size=sizes, font_size=10, font_weight='bold',
                        arrowsize=15, edge_color='gray', width=1.5)
                    
                    # Convert plot to image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    
                    # Display the image
                    st.image(buf, use_container_width=True)
                    
                    # Add legend
                    st.markdown("""
                    **Legend:**
                    - 🟢 Current File
                    - 🔵 Imported Module
                    - ⚪ Module Component
                    """)
                else:
                    st.info("No module dependencies detected in the code.")
                    
                # Add explanation
                st.caption("This graph shows the dependencies between modules and components in your code.")
        
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            return None
        

    def generate_test_cases(self, code):
        """Generate test cases for the provided code."""
        try:
            # Parse the code
            parsed = ast.parse(code)
            
            # Find all function definitions
            functions = [node for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return "No functions found to test."
            
            # Create test file content
            test_code = []
            test_code.append("import unittest")
            
            # Extract additional imports needed from the original code
            imports = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(f"import {name.name}")
                elif isinstance(node, ast.ImportFrom):
                    names = ", ".join(name.name for name in node.names)
                    imports.append(f"from {node.module} import {names}")
                    
            # Add original imports to test file
            for imp in set(imports):
                test_code.append(imp)
            
            # Import the original module (assuming it would be saved as a file)
            test_code.append("\n# Import the module to test")
            test_code.append("# from your_module import *  # Uncomment and adjust this line")
            
            test_code.append("\n\nclass TestFunctions(unittest.TestCase):")
            
            # Generate test methods for each function
            for func in functions:
                func_name = func.name
                # Skip special methods like __init__, etc.
                if func_name.startswith('_') and func_name.endswith('_'):
                    continue
                    
                # Extract docstring if available
                docstring = ast.get_docstring(func)
                
                # Create test method
                test_code.append(f"\n    def test_{func_name}(self):")
                if docstring:
                    test_code.append(f'        """Test {func_name} - {docstring.splitlines()[0]}"""')
                else:
                    test_code.append(f'        """Test {func_name} function"""')
                
                # Generate parameter values for testing
                params = []
                for arg in func.args.args:
                    if arg.arg != 'self':  # Skip self for class methods
                        # Generate appropriate test value based on arg name hints
                        arg_name = arg.arg.lower()
                        if 'num' in arg_name or 'count' in arg_name or 'index' in arg_name:
                            params.append('5')  # A reasonable number for numeric params
                        elif 'str' in arg_name or 'name' in arg_name or 'text' in arg_name:
                            params.append('"test_string"')
                        elif 'list' in arg_name or 'array' in arg_name:
                            params.append('[1, 2, 3]')
                        elif 'dict' in arg_name or 'map' in arg_name:
                            params.append('{"key": "value"}')
                        elif 'bool' in arg_name or 'flag' in arg_name:
                            params.append('True')
                        else:
                            params.append('None')  # Default value
                
                param_str = ', '.join(params)
                
                # Create test assertions
                test_code.append(f"        # Setup - prepare any data needed for the test")
                test_code.append(f"        # Execute the function to test")
                test_code.append(f"        result = {func_name}({param_str})  # Update with appropriate values")
                test_code.append(f"        ")
                test_code.append(f"        # Assert - check the results")
                test_code.append(f"        self.assertIsNotNone(result)  # Replace with appropriate assertion")
                test_code.append(f"        # Add more assertions to test the function thoroughly")
                test_code.append(f"        # self.assertEqual(expected_value, result)")
                
                # Add test for edge cases
                test_code.append(f"\n    def test_{func_name}_edge_cases(self):")
                test_code.append(f'        """Test edge cases for {func_name} function"""')
                test_code.append(f"        # Test with edge cases like None, empty values, etc.")
                test_code.append(f"        # self.assertRaises(ValueError, {func_name}, invalid_param)")
                test_code.append(f"        pass  # Update with actual edge case tests")
            
            # Add main block to run tests
            test_code.append("\n\nif __name__ == '__main__':")
            test_code.append("    unittest.main()")
            
            return "\n".join(test_code)
            
        except SyntaxError as e:
            return f"Couldn't generate tests due to syntax error: {e}"
        except Exception as e:
            return f"Error generating test cases: {e}"



def get_download_link(file_path, link_text):
    """Generate a download link for a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    
    b64_data = base64.b64encode(data.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64_data}" download="{os.path.basename(file_path)}">{link_text}</a>'


    # Main Streamlit app
def main():
    """Main Streamlit application function."""
    # Set page title and configuration
    st.set_page_config(
        page_title="Code Explainer",
        page_icon="🧠",
        layout="wide",
    )
    
    # Main title
    st.title("🧠 AI Code Explainer")
    st.markdown("Using T5 model to explain Python code and analyze its complexity")
    
    # Initialize session state for the explainer
    if 'explainer' not in st.session_state:
        with st.spinner("Initializing Code Explainer..."):
            try:
                # Pass custom model name if specified
                model_name = "Atharva177/T5_new"
                st.session_state.explainer = CodeExplainer(model_name)
            except Exception as e:
                st.error(f"Failed to initialize model: {e}")
                st.stop()
    
    # Text area for code input
    st.subheader("📝 Enter Python Code")
    
    # Add tabs for different input methods
    code_tab1, code_tab2 = st.tabs(["📝 Enter Code", "📁 Upload File"])
    
    # Variable to store the code for analysis
    code = ""
    
    with code_tab1:
        # Sample code for the demo
        default_code = """def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number recursively.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""
        
        code_from_editor = st.text_area("Paste your Python code here", default_code, height=300)
        if code_from_editor:
            code = code_from_editor

    with code_tab2:
        uploaded_file = st.file_uploader("Upload a Python file (.py)", type=["py"])
        if uploaded_file is not None:
            # Read the content of the uploaded file
            try:
                code_content = uploaded_file.getvalue().decode("utf-8")
                st.success(f"Successfully loaded {uploaded_file.name} ({len(code_content.splitlines())} lines)")
                
                # Display a preview of the file content
                with st.expander("Preview file content", expanded=False):
                    st.code(code_content[:1000] + ("..." if len(code_content) > 1000 else ""), language="python")
                
                # Use this content instead of the text area
                code = code_content
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # ESP32 connection
    # In the ESP32 connection section in main(), add:
    with st.expander("Connect to ESP32 OLED Display"):
        st.markdown("Configure connection to an ESP32 with OLED display to show explanations")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            esp32_ip = st.text_input("ESP32 IP Address")
        with col2:
            if st.button("Connect"):
                if esp32_ip:
                    if st.session_state.explainer.esp32.setup_connection(esp32_ip):
                        # Add test button when connected
                        if st.button("Send Test Message"):
                            st.session_state.explainer.esp32.test_display()
                else:
                    st.warning("Please enter an IP address")
    
    # Explanation generation
    if st.button("Explain Code", type="primary", disabled=not code.strip()):
        with st.container():
            # Generate explanation
            explanation = st.session_state.explainer.explain_code(code)
            
            # Show explanation
            st.header("📝 AI-Generated Explanation")
            st.markdown(explanation)
            
            # Show complexity analysis
            st.header("📊 Code Complexity Analysis")
            complexity_metrics = st.session_state.explainer.analyze_code_complexity(code)
            
            # Show code highlighting analysis
            st.header("🔍 Important Code Elements")
            important_elements, highlighted_code = st.session_state.explainer.highlight_important_code_elements(code, explanation)
            
            if important_elements:
                # Show top elements in a table
                st.subheader("Most Referenced Code Elements")
                element_data = {"Element": [], "Mentions": [], "Significance": []}
                
                for element, count in important_elements[:10]:  # Top 10 elements
                    significance = "High" if count >= 5 else "Medium" if count >= 2 else "Low"
                    element_data["Element"].append(element)
                    element_data["Mentions"].append(count)
                    element_data["Significance"].append(significance)
                
                # Convert to DataFrame for table display
                st.table(pd.DataFrame(element_data))
                
                # Show highlighted code
                st.subheader("Code with Highlighted Important Elements")
                st.code(highlighted_code, language="python")
            else:
                st.info("No important code elements identified")
                
            # Show interactive visualization
            st.header("📊 Code Visualization")
            st.session_state.explainer.visualize_code_structure(code)
            
            # Add test case generation section
            st.header("🧪 Unit Test Generation")
            with st.expander("Generate Unit Tests", expanded=True):
                st.write("Automatically generate unit tests for functions in the code.")
                
                test_cases = st.session_state.explainer.generate_test_cases(code)
                
                # Display generated test cases
                st.subheader("Generated Test Cases")
                st.code(test_cases, language="python")
                
                # Allow downloading the test file
                test_file_path = "generated_tests.py"
                with open(test_file_path, "w", encoding="utf-8") as f:
                    f.write(test_cases)
                
                st.download_button(
                    label="Download Test File",
                    data=test_cases,
                    file_name="generated_tests.py",
                    mime="text/plain"
                )
                
                # Add test generation explanation
                st.info("""
                **About the generated tests:**
                - These are unittest framework tests for the functions in your code
                - You may need to adjust imports and parameter values
                - Check the assertions and update them for your specific use case
                - Test both normal operation and edge cases
                """)
            
            # Audio version (if available)
            if HAS_GTTS:
                st.header("🔊 Audio Explanation")
                st.session_state.explainer.create_audio_player(explanation)
            
            # Send to ESP32 if connected
            if st.session_state.explainer.esp32.connected:
                if st.button("Send to ESP32 OLED", key="send_to_esp32"):
                    st.session_state.explainer.esp32.send_explanation(explanation)
            
            # Save to file and provide download link
            output_path = "explanation.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("CODE:\n\n")
                f.write(code)
                f.write("\n\nEXPLANATION:\n\n")
                f.write(explanation)
                
                # Add complexity analysis to the output file
                f.write("\n\nCODE COMPLEXITY ANALYSIS:\n")
                for key, value in complexity_metrics.items():
                    f.write(f"{key}: {value}\n")
                
                # Add important elements to the output file
                f.write("\n\nIMPORTANT CODE ELEMENTS:\n")
                for element, count in important_elements[:10]:
                    f.write(f"{element}: mentioned {count} times\n")
                    
                # Add generated unit tests to the output file
                f.write("\n\nGENERATED UNIT TESTS:\n")
                f.write(test_cases)
            
            # Provide download link
            st.markdown("### 📥 Download Results")
            st.markdown(get_download_link(output_path, "Download Explanation as Text File"), unsafe_allow_html=True)
    
    # Show app information in sidebar
    with st.sidebar:
        st.header("About")
        st.write("This tool uses a fine-tuned T5 model to explain Python code.")
        st.write("It analyzes code complexity and identifies important elements in your code.")
        
        st.header("Features")
        st.markdown("""
        - 📝 AI-powered code explanation
        - 📊 Code complexity metrics
        - 🔍 Code element highlighting
        - 📊 Interactive code visualization
        - 🧪 Automated unit test generation
        - 🔊 Text-to-speech (if gTTS is installed)
        - 📡 ESP32 OLED display integration
        - 📥 Downloadable explanation
        """)
        
        st.header("Model Information")
        st.info(f"Using model: {st.session_state.explainer.model_name}")
        st.write(f"Running on: {st.session_state.explainer.device}")
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("Created as part of NLP PBL")

    # Run the app
if __name__ == "__main__":
    main()