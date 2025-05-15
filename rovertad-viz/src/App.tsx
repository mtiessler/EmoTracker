import React, { useState, useEffect, useCallback } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';
import Card from 'react-bootstrap/Card';
import Spinner from 'react-bootstrap/Spinner';
import Icon from '@mdi/react';
import { mdiAlertCircleOutline } from '@mdi/js';

import FileUpload from './components/FileUpload';
import PlotContainer from './components/PlotContainer';
import ControlPanel from './components/ControlPanel.tsx'; // Ensure correct extension if it's TSX
import { LoadedData, WordSenses, VizType } from './types'; // Import SenseInfo if needed

import './App.scss';

function App() {
    const [jsonData, setJsonData] = useState<LoadedData | null>(null);
    const [wordList, setWordList] = useState<string[]>([]);
    const [selectedWords, setSelectedWords] = useState<string[]>([]);
    const [senseList, setSenseList] = useState<string[]>([]);
    const [selectedSenseId, setSelectedSenseId] = useState<string>('');
    const [vizType, setVizType] = useState<VizType>('2D-VAD');
    const [error, setError] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);
    const [loadedFileName, setLoadedFileName] = useState<string>('');

    const handleFileLoaded = useCallback((data: LoadedData | null, fileName: string): void => {
        setLoading(true);
        setError('');
        setJsonData(null);
        setWordList([]);
        setSelectedWords([]);
        setSenseList([]);
        setSelectedSenseId('');
        setLoadedFileName('');
        setVizType('2D-VAD');

        setTimeout(() => {
            if (data) {
                try {
                    const words = Object.keys(data).sort((a, b) => a.localeCompare(b));
                    if (words.length === 0) {
                        throw new Error("JSON file contains no word entries.");
                    }
                    const firstWord = words[0];
                    const firstWordData = data[firstWord];
                    if (!firstWordData?.temporal_vad?.x || !firstWordData?.temporal_vad?.v || !firstWordData?.temporal_vad?.a || !firstWordData?.temporal_vad?.d || !firstWordData?.senses) {
                        throw new Error("Data structure mismatch. Check 'temporal_vad' (with x,v,a,d arrays) and 'senses'.");
                    }
                    setJsonData(data);
                    setWordList(words);
                    setSelectedWords([firstWord]);
                    setLoadedFileName(fileName);
                } catch (err: unknown) {
                    let message = "Failed to process file.";
                    if (err instanceof Error) { message = err.message; }
                    else if (typeof err === 'string') { message = err; }
                    setError(message);
                    setJsonData(null);
                }
            } else {
                setError(`Failed to load or parse file: ${fileName}. Check file format and console.`);
                setJsonData(null);
            }
            setLoading(false);
        }, 500);

    }, []);

    const handleWordChange = (values: string[]): void => {
        setSelectedWords(values);
        setSelectedSenseId('');
    };

    const handleSenseChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
        setSelectedSenseId(event.target.value);
    };

    const handleVizChange = (type: VizType): void => {
        setVizType(type);
    };

    useEffect(() => {
        const firstWord = selectedWords.length === 1 ? selectedWords[0] : null;
        if (jsonData && firstWord && jsonData[firstWord]?.senses) {
            const senseKeys = Object.keys(jsonData[firstWord].senses);
            senseKeys.sort((a, b) => a.localeCompare(b));
            setSenseList(senseKeys);
        } else {
            setSenseList([]);
        }
        if (selectedWords.length !== 1) {
            setSelectedSenseId('');
        }
    }, [jsonData, selectedWords]);

    const senseDataForPlot: WordSenses | null = (jsonData && selectedWords.length === 1 && jsonData[selectedWords[0]])
        ? jsonData[selectedWords[0]].senses
        : null;


    const renderVisualizationArea = () => {
        if (loading && !jsonData) {
            return (
                <Card className="placeholder-card h-100">
                    <Card.Header>Status</Card.Header>
                    <Card.Body>
                        <Spinner animation="border" role="status"><span className="visually-hidden">Loading...</span></Spinner>
                        <p className="mt-3 mb-0">Processing file...</p>
                    </Card.Body>
                    <Card.Footer className="plot-card-footer">
                        <div className="sense-info-placeholder">&nbsp;</div>
                    </Card.Footer>
                </Card>
            );
        }

        return (
            <PlotContainer
                vizType={vizType}
                jsonData={jsonData}
                selectedWords={selectedWords}
                senseData={senseDataForPlot}
                selectedSenseId={selectedSenseId}
            />
        );
    };


    return (
        // Using py-4 py-md-5 for responsive vertical padding
        <Container fluid="lg" className="app-container py-4 py-md-5">
            {/* Title uses h1 but styled via .app-title */}
            <h1 className="app-title">EMOTracker</h1>

            {error && (
                <Row className="justify-content-center mb-4">
                    <Col xs={12} md={10} lg={8}>
                        <Alert variant="danger" onClose={() => setError('')} dismissible className="app-alert d-flex align-items-center shadow-sm">
                            <Icon path={mdiAlertCircleOutline} size={1.2} className="me-3 flex-shrink-0" />
                            <div>
                                <Alert.Heading as="h6" className="mb-1">Error Loading Data</Alert.Heading>
                                <p className="mb-0 small">{error}</p>
                            </div>
                        </Alert>
                    </Col>
                </Row>
            )}

            {/* Main content area uses flex-grow to push footer */}
            <div className="main-content-area">
                {/* Row uses h-100 to attempt filling vertical space if needed by children */}
                <Row className="h-100">
                    {/* Controls column with responsive margin-bottom */}
                    <Col md={4} lg={3} className="controls-column mb-4 mb-md-0">
                        <div className="file-upload-wrapper mb-3">
                            <FileUpload onFileLoaded={handleFileLoaded} currentFileName={loadedFileName} />
                        </div>
                        {loading && !jsonData && (
                            <div className="loading-indicator mb-3">
                                <Spinner animation="border" role="status" size="sm" className="me-2">
                                    <span className="visually-hidden">Loading...</span>
                                </Spinner>
                                <span>Processing file...</span>
                            </div>
                        )}

                        <ControlPanel
                            show={!!jsonData && !loading}
                            wordList={wordList}
                            selectedWords={selectedWords}
                            handleWordChange={handleWordChange}
                            senseList={senseList}
                            selectedSenseId={selectedSenseId}
                            handleSenseChange={handleSenseChange}
                            vizType={vizType}
                            handleVizChange={handleVizChange}
                            loading={loading}
                        />

                        {!loading && !jsonData && loadedFileName && error && (
                            <Card className="mt-3 error-placeholder-controls">
                                <Card.Body className="text-center text-muted">
                                    File processed, but contained errors. See message above.
                                </Card.Body>
                            </Card>
                        )}
                    </Col>

                    {/* Visualization column taking remaining space */}
                    <Col md={8} lg={9} className="visualization-column d-flex flex-column">
                        {renderVisualizationArea()}
                    </Col>
                </Row>
            </div>

            {/* Footer with specific class */}
            <footer className="app-footer">
                © EMOTracker - Max Tiessler - {new Date().getFullYear()}
            </footer>
        </Container>
    );
}

export default App;