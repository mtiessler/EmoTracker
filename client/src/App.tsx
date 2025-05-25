import React, { useState, useCallback } from 'react';
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
import ControlPanel from './components/ControlPanel';
import { LoadedData, WordSenses, VizType, PredictedVadPoint} from './types';

import './App.scss';

const FIXED_PREDICT_FROM_YEAR = 2010;

function App() {
    const [jsonData, setJsonData] = useState<LoadedData | null>(null);
    const [wordList, setWordList] = useState<string[]>([]);
    const [selectedWords, setSelectedWords] = useState<string[]>([]);
    const [senseList, setSenseList] = useState<string[]>([]);
    const [rawSensesForSelectedWord, setRawSensesForSelectedWord] = useState<WordSenses | null>(null);
    const [selectedSenseId, setSelectedSenseId] = useState<string>('');
    const [vizType, setVizType] = useState<VizType>('2D-VAD');
    const [error, setError] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);
    const [loadedFileName, setLoadedFileName] = useState<string>('');

    const [isPredicting, setIsPredicting] = useState<boolean>(false);
    const [predictionError, setPredictionError] = useState<string | null>(null);
    const [predictedVadSeries, setPredictedVadSeries] = useState<PredictedVadPoint[] | null>(null);


    const handleFileLoaded = useCallback((data: LoadedData | null, fileName: string): void => {
        setLoading(true);
        setError('');
        setJsonData(null);
        setWordList([]);
        setSelectedWords([]);
        setSenseList([]);
        setRawSensesForSelectedWord(null);
        setSelectedSenseId('');
        setLoadedFileName('');
        setVizType('2D-VAD');
        setPredictedVadSeries(null);
        setPredictionError(null);

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
                    if (data[firstWord]?.senses) {
                        setRawSensesForSelectedWord(data[firstWord].senses);
                        setSenseList(Object.keys(data[firstWord].senses).sort((a,b) => a.localeCompare(b)));
                    }
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

    const handleWordChange = useCallback((values: string[]): void => {
        setSelectedWords(values);
        setSelectedSenseId('');
        setRawSensesForSelectedWord(null);
        setSenseList([]);
        setPredictedVadSeries(null);
        setPredictionError(null);

        if (values.length === 1 && jsonData) {
            const word = values[0];
            const wordData = jsonData[word];
            if (wordData && wordData.senses) {
                setRawSensesForSelectedWord(wordData.senses);
                setSenseList(Object.keys(wordData.senses).sort((a,b) => a.localeCompare(b)));
            }
        }
    }, [jsonData]);

    const handleSenseChange = useCallback((event: React.ChangeEvent<HTMLSelectElement>): void => {
        setSelectedSenseId(event.target.value);
        setPredictedVadSeries(null);
        setPredictionError(null);
    }, []);

    const handleVizChange = useCallback((type: VizType): void => {
        setVizType(type);
        if (!['2D-V', '2D-A', '2D-D', '2D-VAD', '3D', 'LSTM-Forecast'].includes(type) || selectedWords.length !== 1) {
            setPredictedVadSeries(null);
            setPredictionError(null);
        }
    }, [selectedWords]);


    const handlePredict = useCallback(async (targetYear: number) => {
        if (selectedWords.length !== 1 || !jsonData) {
            setPredictionError("Please select a single word to run a forecast.");
            setIsPredicting(false);
            return;
        }
        const wordToPredict = selectedWords[0];

        setIsPredicting(true);
        setPredictionError(null);
        setPredictedVadSeries(null);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    word: wordToPredict,
                    predict_from_year: FIXED_PREDICT_FROM_YEAR,
                    predict_until_year: targetYear
                }),
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({error: "Server error or non-JSON response"}));
                throw new Error(errData.error || `HTTP error ${response.status}`);
            }

            const result = await response.json();
            if (result.predictions && Array.isArray(result.predictions)) {
                setPredictedVadSeries(result.predictions);
            } else {
                throw new Error("Invalid prediction format received from server.");
            }

        } catch (err) {
            console.error("Prediction API call failed:", err);
            setPredictionError(err instanceof Error ? err.message : "Prediction failed.");
        } finally {
            setIsPredicting(false);
        }
    }, [selectedWords, jsonData]);


    const renderVisualizationArea = () => {
        if (loading && !jsonData) {
            return (
                <Card className="placeholder-card h-100">
                    <Card.Header>Status</Card.Header>
                    <Card.Body className="d-flex flex-column justify-content-center align-items-center">
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
                senseData={rawSensesForSelectedWord}
                selectedSenseId={selectedSenseId}
                predictedVadSeries={predictedVadSeries}
                isPredicting={isPredicting}
                predictionError={predictionError}
            />
        );
    };


    return (
        <Container fluid="lg" className="app-container py-4 py-md-5">
            <div className="hero-section">
                <div className="title-container">
                    <h1 className="app-title">
                        <span className="title-gradient">EmoTracker</span>
                    </h1>
                    <p className="app-subtitle">
                        Track how words have evolved over time, forecast the future
                    </p>
                </div>
            </div>

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

            <div className="main-content-area">
                <Row className="h-100">
                    <Col md={4} lg={3} className="controls-column mb-4 mb-md-0">
                        <div className="file-upload-wrapper mb-3">
                            <FileUpload onFileLoaded={handleFileLoaded} currentFileName={loadedFileName} />
                        </div>
                        {loading && !jsonData && !error && (
                            <div className="loading-indicator mb-3 text-muted d-flex align-items-center">
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
                            onPredict={handlePredict}
                            isPredicting={isPredicting}
                        />

                        {!loading && !jsonData && loadedFileName && error && (
                            <Card className="mt-3 error-placeholder-controls">
                                <Card.Body className="text-center text-muted">
                                    File processed, but contained errors. See message above.
                                </Card.Body>
                            </Card>
                        )}
                    </Col>

                    <Col md={8} lg={9} className="visualization-column d-flex flex-column">
                        {renderVisualizationArea()}
                    </Col>
                </Row>
            </div>

            <footer className="app-footer">
                © EMOTracker - Max Tiessler - {new Date().getFullYear()}
            </footer>
        </Container>
    );
}

export default App;