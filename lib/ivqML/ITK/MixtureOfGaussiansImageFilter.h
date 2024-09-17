// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__MixtureOfGaussiansImageFilter__h__
#define __ivqML__ITK__MixtureOfGaussiansImageFilter__h__

#include <functional>
#include <itkImage.h>
#include <itkImageToImageFilter.h>

namespace ivqML
{
  namespace ITK
  {
    /**
     */
    template< class _TInImage, class _TLabel = unsigned char, class _TReal = float >
    class MixtureOfGaussiansImageFilter
      : public itk::ImageToImageFilter< _TInImage, itk::Image< _TLabel, _TInImage::ImageDimension > >
    {
    public:
      using TInImage  = _TInImage;
      using TLabel    = _TLabel;
      using TReal     = _TReal;
      using TOutImage = itk::Image< TLabel, TInImage::ImageDimension >;

      using Self         = MixtureOfGaussiansImageFilter;
      using Superclass   = itk::ImageToImageFilter< TInImage, TOutImage >;
      using Pointer      = itk::SmartPointer< Self >;
      using ConstPointer = itk::SmartPointer< const Self >;

      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TDebug
        =
        std::function< bool( const TReal&, const unsigned long long& ) >;

    public:
      itkNewMacro( Self );
      itkTypeMacro(
        ivqML::ITK::MixtureOfGaussiansImageFilter, itk::ImageToImageFilter
        );

      itkGetStringMacro( InitMethod );
      itkSetStringMacro( InitMethod );

      itkGetConstMacro( NumberOfMeans, unsigned long long );
      itkSetMacro( NumberOfMeans, unsigned long long );

      itkGetConstMacro( Means, TMatrix );
      itkGetConstMacro( Covariances, TMatrix );

    public:
      void SetDebug( TDebug d );

    protected:
      MixtureOfGaussiansImageFilter( );
      virtual ~MixtureOfGaussiansImageFilter( ) override = default;

      virtual void GenerateData( ) override;

    private:
      MixtureOfGaussiansImageFilter( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    protected:
      std::string m_InitMethod { "++" };
      unsigned long long m_NumberOfMeans { 2 };
      TMatrix m_Means;
      TMatrix m_Covariances;

      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#  include <ivqML/ITK/MixtureOfGaussiansImageFilter.hxx>
#endif // ITK_MANUAL_INSTANTIATION

#endif // __ivqML__ITK__MixtureOfGaussiansImageFilter__h__

// eof - $RCSfile$
