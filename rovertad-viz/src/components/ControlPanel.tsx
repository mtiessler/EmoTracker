import React, { useState } from 'react';
import Card from 'react-bootstrap/Card';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Icon from '@mdi/react';
import { MultiValue } from 'react-select';
import {
    mdiTuneVariant,
    mdiFormatLetterMatches,
    mdiSitemapOutline,
    mdiChartTimelineVariant,
    mdiCalendarClock,
    mdiRocketLaunchOutline
} from '@mdi/js';

import WordSelector from './WordSelector';
import SenseSelector from './SenseSelector';
import VizControl from './VizControl';
import { VizType, OptionType } from '../types';

import '../styles/ControlPanel.scss';

interface ControlsPanelProps {
    show: boolean;
    wordList: string[];
    selectedWords: string[];
    handleWordChange: (selectedWordValues: string[]) => void;
    senseList: string[];
    selectedSenseId: string;
    handleSenseChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
    vizType: VizType;
    handleVizChange: (type: VizType) => void;
    loading: boolean;
    onPredict: (targetYear: number) => void;
    isPredicting: boolean;
}

const ControlsPanel: React.FC<ControlsPanelProps> = ({
                                                         show,
                                                         wordList,
                                                         selectedWords,
                                                         handleWordChange,
                                                         senseList,
                                                         selectedSenseId,
                                                         handleSenseChange,
                                                         vizType,
                                                         handleVizChange,
                                                         loading,
                                                         onPredict,
                                                         isPredicting
                                                     }) => {

    const [targetYearForPrediction, setTargetYearForPrediction] = useState<string>('2040');

    if (!show) {
        return null;
    }

    const isMultiWord = selectedWords.length > 1;
    const isSingleWordSelected = selectedWords.length === 1;
    const isSenseDisabled = isMultiWord || !selectedWords[0] || senseList.length === 0 || loading;

    const showPredictionControls = isSingleWordSelected &&
        ['2D-V', '2D-A', '2D-D', '2D-VAD', '3D'].includes(vizType);

    const handleReactSelectWordChange = (
        selectedOptions: MultiValue<OptionType>
    ) => {
        const selectedValues = selectedOptions ? selectedOptions.map(option => option.value) : [];
        handleWordChange(selectedValues);
    };

    const handlePredictClick = () => {
        const year = parseInt(targetYearForPrediction, 10);
        if (!isNaN(year) && year > 2010) {
            onPredict(year);
        } else {
            alert("Please enter a valid target year greater than 2010.");
        }
    };

    const predictionYearOptions = [];
    for (let year = 2015; year <= 2060; year += 5) {
        predictionYearOptions.push(year);
    }

    return (
        <Card className="controls-panel-card shadow-sm">
            <Card.Header>
                <Icon path={mdiTuneVariant} size={0.9} className="header-icon" />
                Controls
            </Card.Header>
            <Card.Body>
                <div className="control-section word-control">
                    <div className="control-label-wrapper">
                        <Icon path={mdiFormatLetterMatches} size={0.8} />
                        <Form.Label htmlFor="react-select-word-input">Select Word(s)</Form.Label>
                    </div>
                    <WordSelector
                        id="react-select-word-input"
                        words={wordList}
                        selectedWords={selectedWords}
                        onChange={handleReactSelectWordChange}
                        disabled={loading || isPredicting}
                    />
                </div>

                <div className="control-section sense-control">
                    <div className="control-label-wrapper">
                        <Icon path={mdiSitemapOutline} size={0.8} />
                        <Form.Label htmlFor="sense-select" className={isSenseDisabled ? 'text-muted' : ''}>
                            Select Sense {isMultiWord ? '(Disabled for Multi-Word)' : '(Optional)'}
                        </Form.Label>
                    </div>
                    <SenseSelector
                        id="sense-select"
                        senses={senseList}
                        selectedSenseId={selectedSenseId}
                        onChange={handleSenseChange}
                        disabled={isSenseDisabled || isPredicting || vizType === 'LSTM-Forecast'}
                    />
                </div>

                <div className="control-section viz-control">
                    <div className="control-label-wrapper">
                        <Icon path={mdiChartTimelineVariant} size={0.8} />
                        <Form.Label className={(loading || isPredicting) ? 'text-muted' : ''}>
                            Visualization Type
                        </Form.Label>
                    </div>
                    <VizControl
                        selectedViz={vizType}
                        onChange={handleVizChange}
                        disabled={loading || isPredicting}
                    />
                </div>

                {showPredictionControls && (
                    <div className="control-section prediction-control mt-3 pt-3 border-top">
                        <div className="control-label-wrapper">
                            <Icon path={mdiCalendarClock} size={0.8} />
                            <Form.Label htmlFor="target-year-select">
                                Forecast Target Year
                            </Form.Label>
                        </div>
                        <div className="d-flex">
                            <Form.Select
                                id="target-year-select"
                                value={targetYearForPrediction}
                                onChange={(e) => setTargetYearForPrediction(e.target.value)}
                                disabled={loading || isPredicting}
                                className="me-2"
                                size="sm"
                            >
                                {predictionYearOptions.map(year => (
                                    <option key={year} value={year}>
                                        {year}
                                    </option>
                                ))}
                            </Form.Select>
                            <Button
                                variant="outline-success"
                                onClick={handlePredictClick}
                                disabled={loading || isPredicting}
                                size="sm"
                                className="predict-button d-flex align-items-center"
                            >
                                <Icon path={mdiRocketLaunchOutline} size={0.7} className="me-1" />
                                Predict
                            </Button>
                        </div>
                        <Form.Text className="text-muted d-block mt-1">
                            Uses data up to 2010 to forecast for the selected target year.
                        </Form.Text>
                    </div>
                )}
            </Card.Body>
        </Card>
    );
};

export default ControlsPanel;