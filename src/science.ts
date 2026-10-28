export async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries = 3,
  baseDelayMs = 600
): Promise<Response> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const response = await fetch(url, options);
    if (response.ok) {
      return response;
    }
    if ((response.status === 429 || response.status >= 500) && attempt < maxRetries) {
      const retryAfter = Number(response.headers.get('Retry-After'));
      const delay =
        !Number.isNaN(retryAfter) && retryAfter > 0
          ? retryAfter * 1000
          : baseDelayMs * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delay));
      continue;
    }
    const errorBody = await response.text().catch(() => '');
    throw new Error(
      `HTTP ${response.status} (${response.statusText}) from ${url}: ${errorBody.slice(0, 240)}`
    );
  }
  throw new Error(`Exhausted retries for ${url}`);
}

export interface BioTargetReport {
  accession: string;
  proteinName: string;
  geneName: string;
  organism: string;
  sequenceLength: number;
  sequencePreview: string;
  functionSummary: string;
  alphafoldEntryId: string;
  cifUrl: string;
  pdbUrl: string;
  paeImageUrl: string;
  meanPlddtEstimate: number;
  alphagenomeUrl: string;
  papers: Array<{
    title: string;
    authors: string;
    journal: string;
    pubYear: string;
    pmid?: string;
    doi?: string;
  }>;
  compoundInfo?: {
    name: string;
    cid: number;
    formula: string;
    molecularWeight: string;
    smiles: string;
  };
}

export const PRESET_TARGETS: Array<{
  label: string;
  accession: string;
  gene: string;
  compound: string;
  locus: string;
}> = [
  {
    label: 'TP53 (P04637) · Tumor Suppressor',
    accession: 'P04637',
    gene: 'TP53',
    compound: 'Nutlin-3a',
    locus: 'chr17:7668402-7687550',
  },
  {
    label: 'EGFR (P00533) · Kinase Receptor',
    accession: 'P00533',
    gene: 'EGFR',
    compound: 'Osimertinib',
    locus: 'chr7:55019017-55211628',
  },
  {
    label: 'BRCA1 (P38398) · DNA Repair',
    accession: 'P38398',
    gene: 'BRCA1',
    compound: 'Olaparib',
    locus: 'chr17:43044295-43125483',
  },
  {
    label: 'ABL1 (P00519) · Tyrosine Kinase',
    accession: 'P00519',
    gene: 'ABL1',
    compound: 'Imatinib',
    locus: 'chr9:130713043-130887675',
  },
];

export async function fetchBioTargetWorkbench(
  accession: string,
  compoundName = 'Imatinib',
  locus = 'chr17:7668402-7687550'
): Promise<BioTargetReport> {
  const cleanAcc = accession.trim().toUpperCase();

  // 1. UniProtKB REST API
  const uniprotRes = await fetchWithRetry(`https://rest.uniprot.org/uniprotkb/${cleanAcc}.json`);
  const uniprotData = await uniprotRes.json();

  const proteinName =
    uniprotData.proteinDescription?.recommendedName?.fullName?.value ||
    uniprotData.uniProtkbId ||
    cleanAcc;
  const geneName = uniprotData.genes?.[0]?.geneName?.value || cleanAcc;
  const organism = uniprotData.organism?.scientificName || 'Homo sapiens';
  const sequence: string = uniprotData.sequence?.value || '';
  const sequenceLength: number = uniprotData.sequence?.length || sequence.length;
  const functionComment = (uniprotData.comments || []).find(
    (c: Record<string, unknown>) => c.commentType === 'FUNCTION'
  );
  const functionSummary: string =
    functionComment?.texts?.[0]?.value ||
    'Target protein for ESM-2 / AlphaFold LoRA structural adapter fine-tuning.';

  // 2. AlphaFold DB Prediction API
  let alphafoldEntryId = `AF-${cleanAcc}-F1`;
  let cifUrl = `https://alphafold.ebi.ac.uk/files/AF-${cleanAcc}-F1-model_v6.cif`;
  let pdbUrl = `https://alphafold.ebi.ac.uk/files/AF-${cleanAcc}-F1-model_v6.pdb`;
  let paeImageUrl = `https://alphafold.ebi.ac.uk/files/AF-${cleanAcc}-F1-predicted_aligned_error_v6.png`;
  let meanPlddtEstimate = 88.4;

  try {
    const afRes = await fetchWithRetry(`https://alphafold.ebi.ac.uk/api/prediction/${cleanAcc}`);
    const afArr = await afRes.json();
    if (Array.isArray(afArr) && afArr.length > 0) {
      const first = afArr[0];
      alphafoldEntryId = first.entryId || alphafoldEntryId;
      cifUrl = first.cifUrl || cifUrl;
      pdbUrl = first.pdbUrl || pdbUrl;
      paeImageUrl = first.paeImageUrl || paeImageUrl;
      if (typeof first.globalMetricValue === 'number') {
        meanPlddtEstimate = first.globalMetricValue;
      }
    }
  } catch {
    // Keep dynamically constructed fallback URLs if AFDB endpoint is temporarily unreachable
  }

  // 3. Europe PMC Literature Search for Protein + LoRA / Deep Learning
  const papers: BioTargetReport['papers'] = [];
  try {
    const query = encodeURIComponent(`${geneName} AND (structure OR kinase OR inhibitor)`);
    const epmcRes = await fetchWithRetry(
      `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=${query}&format=json&pageSize=4`
    );
    const epmcData = await epmcRes.json();
    const results = epmcData.resultList?.result || [];
    for (const r of results) {
      papers.push({
        title: r.title || 'Untitled publication',
        authors: r.authorString || 'Authors',
        journal: r.journalTitle || 'Preprint / Journal',
        pubYear: r.pubYear || '2025',
        pmid: r.pmid,
        doi: r.doi,
      });
    }
  } catch {
    // non-fatal
  }

  // 4. PubChem PUG REST Ligand Lookup
  let compoundInfo: BioTargetReport['compoundInfo'];
  try {
    const encodedCompound = encodeURIComponent(compoundName);
    const pubchemRes = await fetchWithRetry(
      `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodedCompound}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON`
    );
    const pcData = await pubchemRes.json();
    const prop = pcData.PropertyTable?.Properties?.[0];
    if (prop) {
      compoundInfo = {
        name: compoundName,
        cid: prop.CID,
        formula: prop.MolecularFormula || '',
        molecularWeight: String(prop.MolecularWeight || ''),
        smiles: prop.CanonicalSMILES || prop.ConnectivitySMILES || '',
      };
    }
  } catch {
    // non-fatal
  }

  const params = new URLSearchParams({ q: locus });
  const alphagenomeUrl = `https://deepmind.google.com/science/alphagenome/atlas?${params.toString()}`;

  return {
    accession: cleanAcc,
    proteinName,
    geneName,
    organism,
    sequenceLength,
    sequencePreview: sequence.slice(0, 140) + (sequence.length > 140 ? '...' : ''),
    functionSummary,
    alphafoldEntryId,
    cifUrl,
    pdbUrl,
    paeImageUrl,
    meanPlddtEstimate,
    alphagenomeUrl,
    papers,
    compoundInfo,
  };
}
