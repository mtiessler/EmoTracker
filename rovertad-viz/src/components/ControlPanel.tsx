import React from 'react';
import Card from 'react-bootstrap/Card';
import Form from 'react-bootstrap/Form';
import Icon from '@mdi/react';
import { MultiValue } from 'react-select';
import {
    mdiTuneVariant,
    mdiFormatLetterMatches,
    mdiSitemapOutline,
    mdiChartTimelineVariant,
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
                                                     }) => {

    if (!show) {
        return null;
    }

    const isMultiWord = selectedWords.length > 1;
    const isSenseDisabled = isMultiWord || !selectedWords[0] || senseList.length === 0 || loading;

    const handleReactSelectWordChange = (
        selectedOptions: MultiValue<OptionType>
    ) => {
        const selectedValues = selectedOptions ? selectedOptions.map(option => option.value) : [];
        handleWordChange(selectedValues);
    };

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
                        disabled={loading}
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
                        disabled={isSenseDisabled} // Disable based on new logic
                    />
                </div>

                <div className="control-section viz-control">
                    <div className="control-label-wrapper">
                        <Icon path={mdiChartTimelineVariant} size={0.8} />
                        <Form.Label className={loading ? 'text-muted' : ''}>
                            Visualization Type
                        </Form.Label>
                    </div>
                    <VizControl
                        selectedViz={vizType}
                        onChange={handleVizChange}
                        disabled={loading}
                    />
                </div>
            </Card.Body>
        </Card>
    );
};

export default ControlsPanel;