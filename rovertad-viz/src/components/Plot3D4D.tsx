// src/components/Plot3D4D.tsx
import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Font, ColorBar, Annotations } from 'plotly.js';
import Icon from '@mdi/react';
import { mdiAlertCircleOutline } from '@mdi/js';
import { LoadedData } from '../types';
import { getVADDescription, VADDimension } from '../vadUtils'; // Corrected path

import '../styles/Plot3D4D.scss'; // Corrected path

interface Plot3D4DProps {
    selectedWords: string[];
    allWordsData: LoadedData | null;
    senseProportions?: number[] | null;
    is4D?: boolean;
}

const createHoverTextWithDescPlain = (
    time: number[] | undefined,
    v: number[] | undefined,
    a: number[] | undefined,
    d: number[] | undefined,
    word: string
): string[] => {
    if (!time || !v || !a || !d) return [];
    return time.map((t, i) => {
        const vVal = v[i];
        const aVal = a[i];
        const dVal = d[i];
        const vDesc = getVADDescription('Valence', vVal);
        const aDesc = getVADDescription('Arousal', aVal);
        const dDesc = getVADDescription('Dominance', dVal);

        return `<b>Word: ${word} | Year: ${t}</b><br>` +
            `V: ${vVal?.toFixed(3) ?? 'N/A'} ${vDesc ? `(${vDesc})` : ''}<br>` +
            `A: ${aVal?.toFixed(3) ?? 'N/A'} ${aDesc ? `(${aDesc})` : ''}<br>` +
            `D: ${dVal?.toFixed(3) ?? 'N/A'} ${dDesc ? `(${dDesc})` : ''}`;
    });
};

const FONT_FAMILY = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif';
const AXIS_TICK_COLOR = '#6c757d';
const AXIS_LABEL_COLOR = '#6c757d';
const AXIS_LINE_COLOR = '#dee2e6';
const AXIS_GRID_COLOR = '#f8f9fa';
const AXIS_BG_COLOR = '#e9ecef';
const PAPER_BG_COLOR = 'white';
const PLOT_BG_COLOR = 'white';
const TEXT_COLOR = '#212529';
const HOVER_BG_COLOR = 'white';
const HOVER_BORDER_COLOR = '#dee2e6';
const TIME_COLOR_SCALE = 'Blues';
const SENSE_COLOR_SCALE = 'Cividis';
const ANNOTATION_COLOR = '#495057';

const BASE_FONT: Partial<Font> = { family: FONT_FAMILY, size: 11, color: TEXT_COLOR };
const AXIS_TICK_FONT: Partial<Font> = { family: FONT_FAMILY, size: 9, color: AXIS_TICK_COLOR };
const AXIS_TITLE_FONT: Partial<Font> = { family: FONT_FAMILY, size: 10, color: AXIS_LABEL_COLOR };
const COLORBAR_TICK_FONT: Partial<Font> = { family: FONT_FAMILY, size: 9, color: AXIS_TICK_COLOR };
const HOVER_FONT: Partial<Font> = { family: FONT_FAMILY, size: 12.8, color: TEXT_COLOR };
const ANNOTATION_FONT: Partial<Font> = { family: FONT_FAMILY, size: 9, color: ANNOTATION_COLOR };

const traceColors = ['#0d6efd', '#dc3545', '#198754', '#ffc107', '#6f42c1', '#fd7e14', '#20c997', '#6610f2'];

const Plot3D4D: React.FC<Plot3D4DProps> = ({ selectedWords, allWordsData, senseProportions = null, is4D = false }) => {

    const plotMemoData = useMemo(() => {
        const traces: Data[] = [];
        let hasValidData = false;
        const annotations: Partial<Annotations>[] = [];
        let dataRanges = {
            x: [0.5, 0.5] as [number, number],
            y: [0.5, 0.5] as [number, number],
            z: [0.5, 0.5] as [number, number]
        };

        if (!allWordsData || selectedWords.length === 0) {
            return { traces, hasValidData, annotations, dataRanges };
        }

        const canShow4D = is4D && selectedWords.length === 1 && Array.isArray(senseProportions) && senseProportions.length > 0;
        const allV: number[] = [];
        const allA: number[] = [];
        const allD: number[] = [];

        selectedWords.forEach((word, index) => {
            const wordData = allWordsData[word];
            if (!wordData?.temporal_vad?.x || !wordData?.temporal_vad?.v || !wordData?.temporal_vad?.a || !wordData?.temporal_vad?.d) {
                console.warn(`Missing VAD data for word: ${word}`);
                return;
            }
            hasValidData = true;
            const { x: time, v, a, d } = wordData.temporal_vad;
            const traceColor = traceColors[index % traceColors.length];
            const dataLength = Math.min(time.length, v.length, a.length, d.length);
            const vData = v.slice(0, dataLength);
            const aData = a.slice(0, dataLength);
            const dData = d.slice(0, dataLength);
            const timeData = time.slice(0, dataLength);

            allV.push(...vData.filter((val): val is number => typeof val === 'number' && !isNaN(val)));
            allA.push(...aData.filter((val): val is number => typeof val === 'number' && !isNaN(val)));
            allD.push(...dData.filter((val): val is number => typeof val === 'number' && !isNaN(val)));

            const hoverText = createHoverTextWithDescPlain(timeData, vData, aData, dData, word);
            const showTimeColorbar = !canShow4D && selectedWords.length <= 1;

            const colorbarOptions: Partial<ColorBar> | undefined = (canShow4D || showTimeColorbar) ? {
                title: canShow4D ? 'Sense Proportion' : 'Time (Year)',
                tickfont: COLORBAR_TICK_FONT,
                bgcolor: 'rgba(0,0,0,0)',
                outlinecolor: AXIS_LINE_COLOR,
                thickness: 15,
                len: 0.7,
            } : undefined;

            traces.push({
                x: vData,
                y: aData,
                z: dData,
                mode: 'lines+markers',
                type: 'scatter3d',
                name: word,
                text: hoverText,
                hoverinfo: 'text',
                line: {
                    color: traceColor,
                    width: selectedWords.length > 1 ? 2 : 1.5,
                    smoothing: 1.0
                },
                marker: {
                    size: selectedWords.length > 1 ? 3.5 : 4.5,
                    color: canShow4D ? senseProportions?.slice(0, dataLength) : timeData,
                    colorscale: canShow4D ? SENSE_COLOR_SCALE : (showTimeColorbar ? TIME_COLOR_SCALE : undefined),
                    showscale: canShow4D || showTimeColorbar,
                    colorbar: colorbarOptions,
                    opacity: selectedWords.length > 1 ? 0.7 : 0.85
                }
            });
        });

        if (hasValidData) {
            const getMinMax = (vals: number[]): [number, number] => {
                if (vals.length === 0) return [0, 1];
                return [Math.min(...vals), Math.max(...vals)];
            };
            dataRanges.x = getMinMax(allV);
            dataRanges.y = getMinMax(allA);
            dataRanges.z = getMinMax(allD);

            const [xMin, xMax] = dataRanges.x;
            const [yMin, yMax] = dataRanges.y;
            const [zMin, zMax] = dataRanges.z;
            const xOffset = Math.max(0.05, (xMax - xMin) * 0.05);
            const yOffset = Math.max(0.05, (yMax - yMin) * 0.05);
            const zOffset = Math.max(0.05, (zMax - zMin) * 0.05);

            annotations.push(
                { text: 'Pleasant', x: xMax + xOffset, y: yMin, z: zMin, showarrow: false, font: ANNOTATION_FONT, xanchor: 'left', yanchor: 'middle'},
                { text: 'Unpleasant', x: xMin - xOffset, y: yMin, z: zMin, showarrow: false, font: ANNOTATION_FONT, xanchor: 'right', yanchor: 'middle' },
                { text: 'Activated', x: xMin, y: yMax + yOffset, z: zMin, showarrow: false, font: ANNOTATION_FONT, xanchor: 'center', yanchor: 'bottom'},
                { text: 'Calm', x: xMin, y: yMin - yOffset, z: zMin, showarrow: false, font: ANNOTATION_FONT, xanchor: 'center', yanchor: 'top'},
                { text: 'In Control', x: xMin, y: yMin, z: zMax + zOffset, showarrow: false, font: ANNOTATION_FONT, xanchor: 'left', yanchor: 'middle'},
                { text: 'Controlled', x: xMin, y: yMin, z: zMin - zOffset, showarrow: false, font: ANNOTATION_FONT, xanchor: 'left', yanchor: 'middle'},
            );
        }

        return { traces, hasValidData, annotations, dataRanges };

    }, [selectedWords, allWordsData, is4D, senseProportions]);


    if (selectedWords.length === 0) {
        return (
            <div className="plot-placeholder info">
                <h4>Select Word(s)</h4>
                <p>Select one or more words from the controls to view the 3D plot.</p>
            </div>
        );
    }

    if (!plotMemoData.hasValidData) {
        return (
            <div className="plot-placeholder error">
                <Icon path={mdiAlertCircleOutline} size={1.6} className="placeholder-icon" />
                <h4>Data Error</h4>
                <p>No valid VAD data found for the selected word(s).</p>
            </div>
        );
    }

    const commonAxisSettings = {
        backgroundcolor: AXIS_BG_COLOR,
        gridcolor: AXIS_GRID_COLOR,
        showbackground: true,
        zerolinecolor: AXIS_LINE_COLOR,
        tickfont: AXIS_TICK_FONT,
        linecolor: AXIS_LINE_COLOR,
        automargin: true,
    };

    const layout: Partial<Layout> = {
        autosize: true,
        height: undefined,
        width: undefined,
        margin: { l: 0, r: 0, b: 0, t: 0, pad: 4 },
        showlegend: selectedWords.length > 1,
        legend: {
            font: { size: 9 },
            yanchor: "top",
            y: 0.99,
            xanchor: "left",
            x: 0.01,
            bgcolor: 'rgba(255,255,255,0.7)',
            bordercolor: AXIS_LINE_COLOR,
            borderwidth: 1
        },
        scene: {
            xaxis: { ...commonAxisSettings, title: { text: 'Valence (V)', font: AXIS_TITLE_FONT } },
            yaxis: { ...commonAxisSettings, title: { text: 'Arousal (A)', font: AXIS_TITLE_FONT } },
            zaxis: { ...commonAxisSettings, title: { text: 'Dominance (D)', font: AXIS_TITLE_FONT } },
            camera: { eye: { x: 1.6, y: 1.6, z: 1.6 } },
            aspectmode: 'cube',
            annotations: plotMemoData.annotations
        },
        paper_bgcolor: PAPER_BG_COLOR,
        plot_bgcolor: PLOT_BG_COLOR,
        font: BASE_FONT,
        hoverlabel: {
            bgcolor: HOVER_BG_COLOR,
            bordercolor: HOVER_BORDER_COLOR,
            font: HOVER_FONT,
            align: 'left',
            namelength: -1
        }
    };

    return (
        <div className="plot3d-wrapper">
            <Plot
                data={plotMemoData.traces}
                layout={layout}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
                config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'resetCameraDefault3d'] }}
            />
        </div>
    );
}

export default Plot3D4D;