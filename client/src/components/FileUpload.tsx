import React, { useCallback } from 'react';
import { useDropzone, Accept } from 'react-dropzone';
import Card from 'react-bootstrap/Card';
import Icon from '@mdi/react';
import { mdiFileUploadOutline, mdiCheckCircleOutline } from '@mdi/js';
import { LoadedData } from '../types';

import '../styles/FileUpload.scss';

interface FileUploadProps {
    onFileLoaded: (data: LoadedData | null, fileName: string) => void;
    currentFileName?: string;
}

const jsonAccept: Accept = { 'application/json': ['.json'] };

const FileUpload: React.FC<FileUploadProps> = ({ onFileLoaded, currentFileName }) => {
    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length !== 1) {
            console.error("Please drop only one JSON file.");
            onFileLoaded(null, 'Multiple files selected');
            return;
        }
        const file = acceptedFiles[0];
        const reader = new FileReader();

        reader.onabort = () => {
            console.log('file reading was aborted');
            onFileLoaded(null, file.name);
        }
        reader.onerror = (e) => {
            console.error('File reading has failed:', e);
            onFileLoaded(null, file.name);
        }
        reader.onload = () => {
            try {
                const fileContent = reader.result as string;
                if (!fileContent) {
                    throw new Error("File content is empty or could not be read.");
                }
                const jsonData: LoadedData = JSON.parse(fileContent);
                onFileLoaded(jsonData, file.name);
            } catch (e) {
                console.error("Error parsing JSON:", e);
                onFileLoaded(null, file.name);
            }
        };
        reader.readAsText(file);
    }, [onFileLoaded]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: jsonAccept,
        multiple: false,
        noClick: false,
        noKeyboard: true,
    });

    return (
        <Card className="fileupload-card shadow-sm">
            <Card.Body>
                <Card.Title as="h6">Load Dataset</Card.Title>
                <div
                    {...getRootProps()}
                    className={`dropzone ${isDragActive ? 'active' : ''}`}
                >
                    <input {...getInputProps()} />
                    <Icon path={mdiFileUploadOutline} size={1.3} />
                    {isDragActive ?
                        (<p>Drop the JSON file here...</p>) :
                        (<p>Drag & drop file or click to select</p>)
                    }
                    {currentFileName && !isDragActive && (
                        <div className="loaded-file-info text-success d-block mt-2">
                            <Icon path={mdiCheckCircleOutline} size={0.8} className="me-1"/>
                            <small>
                                Loaded: {currentFileName}
                            </small>
                        </div>
                    )}
                </div>
            </Card.Body>
        </Card>
    );
}

export default FileUpload;